"""Análisis transversal del corpus: comparar papers entre sí.

En vez de resumir un paper aislado, toma un lote de resúmenes estructurados ya
generados (compactos, así el lote cabe en el contexto del modelo local) y le
pide al LLM una síntesis comparativa: tendencias, consensos, contradicciones,
huecos abiertos, métodos transferibles y aplicaciones viables, citando cada
afirmación con [n] igual que el chat RAG.
"""

import json
import sqlite3
from dataclasses import dataclass

from . import config, db, llm

DEFAULT_LIMIT = 15
DOSSIER_BUDGET = 24000  # chars (≈6k tokens): deja sitio a instrucciones y respuesta
MIN_PAPERS = 2

SECTIONS = [
    ("panorama", "Panorama"),
    ("tendencias", "Tendencias"),
    ("consensos", "Consensos"),
    ("contradicciones", "Contradicciones"),
    ("huecos", "Huecos abiertos"),
    ("metodos_transferibles", "Métodos transferibles"),
    ("aplicaciones", "Aplicaciones posibles"),
]

SYNTHESIS_SYSTEM = (
    "Eres un analista de literatura científica. Respondes siempre en español, "
    "comparando EXCLUSIVAMENTE los papers proporcionados, sin inventar. Cada "
    "afirmación cita sus papers con el número entre corchetes, p. ej. [1] o [2][5]."
)

SYNTHESIS_PROMPT = """Compara los siguientes papers entre sí y devuelve SOLO un objeto JSON con estas claves:
- "panorama": 3-5 frases sobre el estado del tema según estos papers
- "tendencias": lista de direcciones emergentes o giros recientes (strings)
- "consensos": lista de puntos en los que varios papers coinciden (strings)
- "contradicciones": lista de puntos donde los papers discrepan, citando ambos lados (strings)
- "huecos": lista de preguntas abiertas o aspectos que ningún paper cubre (strings)
- "metodos_transferibles": lista de métodos de un paper aplicables a problemas de otros (strings)
- "aplicaciones": lista de aplicaciones prácticas viables con la evidencia de estos papers, indicando su madurez (strings)

Cada elemento debe citar los papers que lo sustentan con [n]. Si una sección no aplica, devuelve una lista vacía.
{topic_line}
PAPERS:
{dossiers}
"""


@dataclass
class Synthesis:
    id: int | None
    topic: str | None
    sections: dict
    sources: list[dict]  # [{n, paper_id, title, year}]
    model: str | None = None
    created_at: str | None = None


# ---------------------------------------------------------------- selección

def select_papers(conn: sqlite3.Connection, topic: str | None, limit: int) -> list[int]:
    """Ids de papers CON resumen a comparar, por relevancia al tema o recencia.

    Con tema: FTS sobre título/abstract y, si no hay match, sobre el texto
    completo (chunks). Sin tema (o sin matches): los resumidos más recientes.
    """
    if topic:
        rows = conn.execute(
            """SELECT p.id FROM papers_fts f
               JOIN papers p ON p.id = f.rowid
               JOIN summaries s ON s.paper_id = p.id
               WHERE papers_fts MATCH ? ORDER BY rank LIMIT ?""",
            (db.fts_escape(topic), limit),
        ).fetchall()
        if not rows:
            rows = conn.execute(
                """SELECT c.paper_id AS id, MIN(rank) AS best
                   FROM chunks_fts f
                   JOIN chunks c ON c.id = f.rowid
                   JOIN summaries s ON s.paper_id = c.paper_id
                   WHERE chunks_fts MATCH ?
                   GROUP BY c.paper_id ORDER BY best LIMIT ?""",
                (db.fts_escape(topic), limit),
            ).fetchall()
        return [r["id"] for r in rows]
    rows = conn.execute(
        """SELECT s.paper_id AS id FROM summaries s
           JOIN papers p ON p.id = s.paper_id
           ORDER BY p.added_at DESC, p.id DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    return [r["id"] for r in rows]


# ---------------------------------------------------------------- dossiers

def _clip(text: str | None, max_chars: int) -> str:
    text = (text or "").strip()
    return text[:max_chars] + "…" if len(text) > max_chars else text


def build_dossiers(
    conn: sqlite3.Connection, paper_ids: list[int]
) -> tuple[str, list[dict]]:
    """Bloques compactos [n] por paper (desde summaries) + lista de fuentes.

    Corta el lote cuando se agota DOSSIER_BUDGET para no desbordar el contexto.
    """
    placeholders = ",".join("?" * len(paper_ids))
    rows = conn.execute(
        f"""SELECT p.id, p.title, p.year, p.venue,
                   s.summary_md, s.findings, s.method, s.limitations
            FROM papers p JOIN summaries s ON s.paper_id = p.id
            WHERE p.id IN ({placeholders})""",
        paper_ids,
    ).fetchall()
    by_id = {r["id"]: r for r in rows}

    blocks: list[str] = []
    sources: list[dict] = []
    used = 0
    for pid in paper_ids:  # respeta el orden de relevancia
        row = by_id.get(pid)
        if row is None:
            continue
        try:
            hallazgos = json.loads(row["findings"] or "[]")
        except json.JSONDecodeError:
            hallazgos = []
        n = len(sources) + 1
        parts = [f"[{n}] {row['title']} ({row['year'] or 's.f.'}{', ' + row['venue'] if row['venue'] else ''})"]
        if row["summary_md"]:
            parts.append(f"Resumen: {_clip(row['summary_md'], 500)}")
        if hallazgos:
            parts.append("Hallazgos: " + "; ".join(str(h) for h in hallazgos[:5]))
        if row["method"]:
            parts.append(f"Método: {_clip(row['method'], 300)}")
        if row["limitations"]:
            parts.append(f"Limitaciones: {_clip(row['limitations'], 300)}")
        block = "\n".join(parts)
        if used + len(block) > DOSSIER_BUDGET and len(sources) >= MIN_PAPERS:
            break
        blocks.append(block)
        sources.append({"n": n, "paper_id": pid, "title": row["title"], "year": row["year"]})
        used += len(block)
    return "\n\n".join(blocks), sources


# ---------------------------------------------------------------- síntesis

def _normalize(data: dict) -> dict:
    """Garantiza todas las secciones: panorama string, el resto listas de strings."""
    out: dict = {}
    for key, _label in SECTIONS:
        value = data.get(key)
        if key == "panorama":
            if isinstance(value, list):  # el modelo a veces devuelve frases sueltas
                value = " ".join(str(x) for x in value)
            out[key] = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False) if value else ""
            continue
        if not isinstance(value, list):
            value = [value] if value else []
        out[key] = [
            item if isinstance(item, str) else json.dumps(item, ensure_ascii=False)
            for item in value
        ]
    return out


def run(
    conn: sqlite3.Connection, topic: str | None = None, limit: int = DEFAULT_LIMIT
) -> Synthesis:
    paper_ids = select_papers(conn, topic, limit)
    if len(paper_ids) < MIN_PAPERS:
        raise ValueError(
            "se necesitan al menos 2 papers con resumen"
            + (f" que coincidan con «{topic}»" if topic else "")
            + " — ejecuta antes `paperlab summarize`"
        )
    dossiers, sources = build_dossiers(conn, paper_ids)
    topic_line = f"TEMA DE ENFOQUE: {topic}\n" if topic else ""
    data = llm.generate_json(
        SYNTHESIS_PROMPT.format(topic_line=topic_line, dossiers=dossiers),
        system=SYNTHESIS_SYSTEM,
    )
    sections = _normalize(data)
    cur = conn.execute(
        "INSERT INTO syntheses (topic, paper_ids, sections, model) VALUES (?, ?, ?, ?)",
        (
            topic,
            json.dumps([s["paper_id"] for s in sources]),
            json.dumps(sections, ensure_ascii=False),
            config.OLLAMA_MODEL,
        ),
    )
    conn.commit()
    return Synthesis(
        id=cur.lastrowid, topic=topic, sections=sections,
        sources=sources, model=config.OLLAMA_MODEL,
    )


# ---------------------------------------------------------------- lectura

def _sources_for(conn: sqlite3.Connection, paper_ids: list[int]) -> list[dict]:
    placeholders = ",".join("?" * len(paper_ids))
    rows = conn.execute(
        f"SELECT id, title, year FROM papers WHERE id IN ({placeholders})", paper_ids
    ).fetchall()
    by_id = {r["id"]: r for r in rows}
    sources = []
    for n, pid in enumerate(paper_ids, 1):
        row = by_id.get(pid)
        sources.append({
            "n": n, "paper_id": pid,
            "title": row["title"] if row else f"paper #{pid} (borrado)",
            "year": row["year"] if row else None,
        })
    return sources


def get(conn: sqlite3.Connection, synthesis_id: int) -> Synthesis | None:
    row = conn.execute("SELECT * FROM syntheses WHERE id = ?", (synthesis_id,)).fetchone()
    if not row:
        return None
    paper_ids = json.loads(row["paper_ids"])
    return Synthesis(
        id=row["id"], topic=row["topic"], sections=json.loads(row["sections"]),
        sources=_sources_for(conn, paper_ids), model=row["model"],
        created_at=row["created_at"],
    )


def list_all(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """SELECT id, topic, paper_ids, model, created_at
           FROM syntheses ORDER BY id DESC"""
    ).fetchall()
