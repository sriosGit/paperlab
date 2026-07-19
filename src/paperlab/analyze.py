"""Procesamiento (chunks + embeddings), resúmenes, extracción y RAG."""

import json
import sqlite3
from dataclasses import dataclass

import numpy as np

from . import config, db, llm, pdf

EMBED_BATCH = 32
RRF_K = 60


# ---------------------------------------------------------------- procesamiento

def ensure_chunks(conn: sqlite3.Connection) -> int:
    """Trocea papers que aún no tienen chunks (PDF si existe; si no, abstract)."""
    rows = conn.execute(
        """SELECT p.* FROM papers p
           WHERE NOT EXISTS (SELECT 1 FROM chunks c WHERE c.paper_id = p.id)
             AND (p.pdf_path IS NOT NULL OR p.abstract IS NOT NULL)
             AND p.excluded = 0"""
    ).fetchall()
    done = 0
    for row in rows:
        if row["pdf_path"]:
            try:
                text = pdf.extract_text(row["pdf_path"])
            except Exception:  # noqa: BLE001 — PDF corrupto no debe frenar el lote
                text = row["abstract"] or ""
        else:
            text = row["abstract"] or ""
        if not text.strip():
            continue
        chunks = pdf.chunk_text(text, row["title"])
        for seq, chunk in enumerate(chunks):
            conn.execute(
                "INSERT OR IGNORE INTO chunks (paper_id, seq, text) VALUES (?, ?, ?)",
                (row["id"], seq, chunk),
            )
        conn.execute(
            "UPDATE papers SET status = 'indexed' WHERE id = ? AND status != 'summarized'",
            (row["id"],),
        )
        conn.commit()
        done += 1
    return done


def ensure_embeddings(conn: sqlite3.Connection) -> int:
    """Calcula embeddings de los chunks que no los tienen."""
    rows = conn.execute("SELECT id, text FROM chunks WHERE embedding IS NULL").fetchall()
    for i in range(0, len(rows), EMBED_BATCH):
        batch = rows[i : i + EMBED_BATCH]
        vectors = llm.embed([r["text"] for r in batch])
        for row, vec in zip(batch, vectors):
            conn.execute(
                "UPDATE chunks SET embedding = ? WHERE id = ?",
                (db.embedding_to_blob(vec), row["id"]),
            )
        conn.commit()
    return len(rows)


# ---------------------------------------------------------------- búsqueda híbrida

class _VectorIndex:
    """Matriz de embeddings en memoria, recargada cuando cambia el nº de chunks."""

    def __init__(self) -> None:
        self.ids: np.ndarray | None = None
        self.matrix: np.ndarray | None = None
        self._stamp: tuple | None = None

    def _load(self, conn: sqlite3.Connection) -> None:
        stamp = conn.execute(
            "SELECT COUNT(*), COALESCE(MAX(id), 0) FROM chunks WHERE embedding IS NOT NULL"
        ).fetchone()
        stamp = tuple(stamp)
        if stamp == self._stamp:
            return
        rows = conn.execute("SELECT id, embedding FROM chunks WHERE embedding IS NOT NULL").fetchall()
        if not rows:
            self.ids, self.matrix, self._stamp = None, None, stamp
            return
        self.ids = np.array([r["id"] for r in rows])
        m = np.array([db.blob_to_embedding(r["embedding"]) for r in rows], dtype=np.float32)
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.matrix = m / norms
        self._stamp = stamp

    def search(self, conn: sqlite3.Connection, query_vec: list[float], k: int) -> list[int]:
        self._load(conn)
        if self.matrix is None:
            return []
        q = np.asarray(query_vec, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        scores = self.matrix @ q
        top = np.argsort(-scores)[:k]
        return [int(self.ids[i]) for i in top]


_index = _VectorIndex()


def hybrid_search(
    conn: sqlite3.Connection, query: str, k: int = 8, max_per_paper: int = 2
) -> list[sqlite3.Row]:
    """Recuperación híbrida: vectores + FTS5, fusionados con Reciprocal Rank Fusion.

    `max_per_paper` limita cuántos chunks aporta un mismo artículo, para que la
    respuesta sintetice varias fuentes en vez de citar una sola.
    """
    pool = k * 5  # candidatos por ranking; el tope por paper descarta muchos
    vec_ids: list[int] = []
    try:
        vec_ids = _index.search(conn, llm.embed([query])[0], pool)
    except llm.OllamaError:
        pass  # sin Ollama, degradamos a solo FTS
    fts_ids = [
        r["rowid"]
        for r in conn.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
            (db.fts_escape(query), pool),
        ).fetchall()
    ]
    scores: dict[int, float] = {}
    for ranking in (vec_ids, fts_ids):
        for rank, cid in enumerate(ranking):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
    if not scores:
        return []
    ranked_ids = sorted(scores, key=scores.get, reverse=True)
    placeholders = ",".join("?" * len(ranked_ids))
    rows = conn.execute(
        f"""SELECT c.id, c.text, c.paper_id, p.title, p.year, p.source
            FROM chunks c JOIN papers p ON p.id = c.paper_id
            WHERE c.id IN ({placeholders}) AND p.excluded = 0""",
        ranked_ids,
    ).fetchall()
    by_id = {r["id"]: r for r in rows}

    selected: list[sqlite3.Row] = []
    per_paper: dict[int, int] = {}
    overflow: list[sqlite3.Row] = []
    for cid in ranked_ids:
        row = by_id.get(cid)
        if row is None:
            continue
        if per_paper.get(row["paper_id"], 0) < max_per_paper:
            per_paper[row["paper_id"]] = per_paper.get(row["paper_id"], 0) + 1
            selected.append(row)
            if len(selected) == k:
                return selected
        else:
            overflow.append(row)
    # si no hay suficientes papers distintos, se rellena con los mejores descartados
    selected.extend(overflow[: k - len(selected)])
    return selected


# ---------------------------------------------------------------- resúmenes

SUMMARY_SYSTEM = (
    "Eres un asistente de investigación científica. Respondes siempre en español, "
    "con precisión y sin inventar información que no esté en el texto."
)

SUMMARY_PROMPT = """Analiza el siguiente artículo científico y devuelve SOLO un objeto JSON con estas claves:
- "resumen": resumen de 4-6 frases del artículo
- "hallazgos": lista de 3-5 hallazgos clave (strings)
- "metodo": descripción breve del método o enfoque
- "limitaciones": limitaciones mencionadas o evidentes (string)
- "relevancia": para quién o para qué es relevante este trabajo (string)

TÍTULO: {title}
AÑO: {year}

TEXTO:
{text}
"""


def summarize_paper(conn: sqlite3.Connection, paper_id: int) -> dict:
    row = conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
    if not row:
        raise ValueError(f"paper {paper_id} no existe")
    # texto: abstract + primeros chunks del PDF hasta ~5000 tokens (≈20k chars)
    parts = [row["abstract"] or ""]
    for c in conn.execute(
        "SELECT text FROM chunks WHERE paper_id = ? ORDER BY seq LIMIT 8", (paper_id,)
    ).fetchall():
        parts.append(c["text"])
    text = "\n\n".join(p for p in parts if p)[:20000]
    if not text.strip():
        raise ValueError(f"paper {paper_id} no tiene texto (ni abstract ni PDF procesado)")
    data = llm.generate_json(
        SUMMARY_PROMPT.format(title=row["title"], year=row["year"] or "?", text=text),
        system=SUMMARY_SYSTEM,
    )
    conn.execute(
        """INSERT OR REPLACE INTO summaries
           (paper_id, summary_md, findings, method, limitations, relevance, model)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            paper_id,
            str(data.get("resumen", "")),
            json.dumps(data.get("hallazgos", []), ensure_ascii=False),
            str(data.get("metodo", "")),
            str(data.get("limitaciones", "")),
            str(data.get("relevancia", "")),
            config.OLLAMA_MODEL,
        ),
    )
    conn.execute("UPDATE papers SET status = 'summarized' WHERE id = ?", (paper_id,))
    conn.commit()
    return data


def pending_summaries(conn: sqlite3.Connection) -> list[int]:
    return [
        r["id"]
        for r in conn.execute(
            """SELECT p.id FROM papers p
               WHERE NOT EXISTS (SELECT 1 FROM summaries s WHERE s.paper_id = p.id)
                 AND (p.abstract IS NOT NULL
                      OR EXISTS (SELECT 1 FROM chunks c WHERE c.paper_id = p.id))
                 AND p.excluded = 0
               ORDER BY p.id"""
        ).fetchall()
    ]


# ---------------------------------------------------------------- RAG

ASK_SYSTEM = (
    "Eres un asistente de investigación. Respondes en español usando EXCLUSIVAMENTE "
    "las fuentes proporcionadas. Cita cada afirmación con el número de fuente entre "
    "corchetes, p. ej. [1] o [2][3]. Si las fuentes no contienen la respuesta, dilo."
)

ASK_PROMPT = """FUENTES:
{sources}

PREGUNTA: {question}

Responde de forma clara y concisa citando las fuentes con [n]."""


@dataclass
class Answer:
    text: str
    sources: list[dict]  # [{n, paper_id, title, year}]


def ask(conn: sqlite3.Connection, question: str, k: int = 8) -> Answer:
    chunks = hybrid_search(conn, question, k=k)
    if not chunks:
        return Answer(
            text="No hay contenido indexado todavía. Ingresa papers y ejecuta el procesamiento.",
            sources=[],
        )
    seen: dict[int, int] = {}  # paper_id -> n
    sources: list[dict] = []
    blocks: list[str] = []
    for c in chunks:
        if c["paper_id"] not in seen:
            n = len(seen) + 1
            seen[c["paper_id"]] = n
            sources.append(
                {"n": n, "paper_id": c["paper_id"], "title": c["title"], "year": c["year"]}
            )
        n = seen[c["paper_id"]]
        blocks.append(f"[{n}] {c['title']} ({c['year'] or 's.f.'}):\n{c['text'][:2000]}")
    answer = llm.generate(
        ASK_PROMPT.format(sources="\n\n".join(blocks), question=question),
        system=ASK_SYSTEM,
    )
    return Answer(text=answer.strip(), sources=sources)
