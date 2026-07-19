"""Análisis transversal del corpus: comparar papers entre sí.

En vez de resumir un paper aislado, toma un lote de resúmenes estructurados ya
generados (compactos, así el lote cabe en el contexto del modelo local) y le
pide al LLM una síntesis comparativa: tendencias, consensos, contradicciones,
huecos abiertos, métodos transferibles y aplicaciones viables, citando cada
afirmación con [n] igual que el chat RAG.

Para corpus que no caben en una ventana de contexto está el modo map-reduce
(`run_full`): analiza el corpus por lotes con numeración global de citas
(map) y combina los análisis parciales en una síntesis final (reduce,
recursivo si los parciales tampoco caben de una vez).
"""

import json
import math
import re
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field

from . import config, db, llm

DEFAULT_LIMIT = 15
DOSSIER_BUDGET = 24000  # chars (≈6k tokens): deja sitio a instrucciones y respuesta
MIN_PAPERS = 2
REDUCE_GROUP = 6        # análisis parciales por llamada de reducción

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

REGLA OBLIGATORIA DE CITAS: TODO elemento de TODA lista —incluida "tendencias"— y el "panorama" deben terminar citando los papers que los sustentan con [n], p. ej. "más uso de X [2][7]". Un elemento sin cita es inválido: si no puedes citarlo, no lo incluyas. Si una sección no aplica, devuelve una lista vacía.
{topic_line}
PAPERS:
{dossiers}
"""

REDUCE_PROMPT = """Combina los siguientes análisis parciales (cada uno cubre un lote distinto de papers del MISMO corpus) en UNA síntesis global. Devuelve SOLO un objeto JSON con las mismas claves y formato que los parciales: "panorama" (string), "tendencias", "consensos", "contradicciones", "huecos", "metodos_transferibles" y "aplicaciones" (listas de strings).

Reglas:
- Conserva las citas [n] tal cual: cada número identifica un paper concreto y es consistente entre lotes. No renumeres ni inventes citas.
- TODO elemento de TODA lista (incluida "tendencias") y el "panorama" deben llevar al menos una cita [n]; descarta los puntos que llegues sin poder citar.
- Fusiona los puntos duplicados o muy similares sumando sus citas; prioriza los que aparecen en varios lotes.
- Busca contradicciones también ENTRE lotes: afirmaciones de un lote que choquen con las de otro.
- Un hueco solo es global si ningún lote lo cubre; descarta los huecos que otro lote resuelve.
{topic_line}
ANÁLISIS PARCIALES:
{partials}
"""


@dataclass
class Synthesis:
    id: int | None
    topic: str | None
    sections: dict
    sources: list[dict]  # [{n, paper_id, title, year}]
    model: str | None = None
    created_at: str | None = None
    audit: "CitationAudit | None" = None


@dataclass
class CitationAudit:
    """Estado de las citas [n] de una síntesis frente a sus fuentes reales."""

    n_sources: int
    citadas: set[int] = field(default_factory=set)
    fuera_de_rango: list[int] = field(default_factory=list)
    secciones_sin_citas: list[str] = field(default_factory=list)

    @property
    def sin_citar(self) -> int:
        return self.n_sources - len(self.citadas)

    @property
    def cobertura(self) -> float:
        return len(self.citadas) / self.n_sources if self.n_sources else 0.0

    @property
    def ok(self) -> bool:
        return not self.fuera_de_rango and not self.secciones_sin_citas


_CITA = re.compile(r"\[(\d+)\]")


def audit_citations(sections: dict, n_sources: int) -> CitationAudit:
    """Comprueba que las citas apunten a fuentes existentes y que nada quede sin citar.

    No valida que la afirmación se sostenga en el paper citado (eso exigiría
    otra pasada del LLM), pero sí detecta el fallo más grave —citar una fuente
    inexistente— y las secciones que el modelo devolvió sin respaldo alguno.
    """
    audit = CitationAudit(n_sources=n_sources)
    for key, _label in SECTIONS:
        value = sections.get(key)
        if not value:
            continue
        texto = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        nums = [int(n) for n in _CITA.findall(texto)]
        if not nums:
            audit.secciones_sin_citas.append(key)
        for n in nums:
            if 1 <= n <= n_sources:
                audit.citadas.add(n)
            elif n not in audit.fuera_de_rango:
                audit.fuera_de_rango.append(n)
    return audit


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
               WHERE papers_fts MATCH ? AND p.excluded = 0
               ORDER BY rank LIMIT ?""",
            (db.fts_escape(topic), limit),
        ).fetchall()
        if not rows:
            rows = conn.execute(
                """SELECT c.paper_id AS id, MIN(rank) AS best
                   FROM chunks_fts f
                   JOIN chunks c ON c.id = f.rowid
                   JOIN summaries s ON s.paper_id = c.paper_id
                   JOIN papers p ON p.id = c.paper_id
                   WHERE chunks_fts MATCH ? AND p.excluded = 0
                   GROUP BY c.paper_id ORDER BY best LIMIT ?""",
                (db.fts_escape(topic), limit),
            ).fetchall()
        return [r["id"] for r in rows]
    rows = conn.execute(
        """SELECT s.paper_id AS id FROM summaries s
           JOIN papers p ON p.id = s.paper_id
           WHERE p.excluded = 0
           ORDER BY p.added_at DESC, p.id DESC LIMIT ?""",
        (limit,),
    ).fetchall()
    return [r["id"] for r in rows]


# ---------------------------------------------------------------- dossiers

def _clip(text: str | None, max_chars: int) -> str:
    text = (text or "").strip()
    return text[:max_chars] + "…" if len(text) > max_chars else text


def build_dossiers(
    conn: sqlite3.Connection, paper_ids: list[int], start_n: int = 1
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
        n = start_n + len(sources)
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


def _require_min(paper_ids: list[int], topic: str | None) -> None:
    if len(paper_ids) < MIN_PAPERS:
        raise ValueError(
            "se necesitan al menos 2 papers con resumen"
            + (f" que coincidan con «{topic}»" if topic else "")
            + " — ejecuta antes `paperlab summarize`"
        )


def _topic_line(topic: str | None) -> str:
    return f"TEMA DE ENFOQUE: {topic}\n" if topic else ""


def _store(
    conn: sqlite3.Connection, topic: str | None, sources: list[dict], sections: dict
) -> Synthesis:
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
        audit=audit_citations(sections, len(sources)),
    )


def run(
    conn: sqlite3.Connection, topic: str | None = None, limit: int = DEFAULT_LIMIT
) -> Synthesis:
    paper_ids = select_papers(conn, topic, limit)
    _require_min(paper_ids, topic)
    dossiers, sources = build_dossiers(conn, paper_ids)
    data = llm.generate_json(
        SYNTHESIS_PROMPT.format(topic_line=_topic_line(topic), dossiers=dossiers),
        system=SYNTHESIS_SYSTEM,
    )
    return _store(conn, topic, sources, _normalize(data))


# ---------------------------------------------------------------- map-reduce

def _reduce(
    partials: list[dict], topic: str | None, notify: Callable[[str], None]
) -> dict:
    """Combina análisis parciales [{first, last, sections}] en rondas hasta dejar uno."""
    ronda = 0
    while len(partials) > 1:
        ronda += 1
        siguientes: list[dict] = []
        for i in range(0, len(partials), REDUCE_GROUP):
            group = partials[i : i + REDUCE_GROUP]
            if len(group) == 1:
                siguientes.append(group[0])
                continue
            notify(f"reducción (ronda {ronda}): combinando {len(group)} análisis parciales…")
            blocks = "\n\n".join(
                f"### Análisis de los papers [{p['first']}]–[{p['last']}]\n"
                + json.dumps(p["sections"], ensure_ascii=False)
                for p in group
            )
            data = llm.generate_json(
                REDUCE_PROMPT.format(topic_line=_topic_line(topic), partials=blocks),
                system=SYNTHESIS_SYSTEM,
            )
            siguientes.append(
                {"first": group[0]["first"], "last": group[-1]["last"],
                 "sections": _normalize(data)}
            )
        partials = siguientes
    return partials[0]["sections"]


def run_full(
    conn: sqlite3.Connection,
    topic: str | None = None,
    batch_size: int = DEFAULT_LIMIT,
    progress: Callable[[str], None] | None = None,
) -> Synthesis:
    """Síntesis map-reduce de TODO el corpus (o de todo lo que coincida con el tema)."""
    notify = progress or (lambda msg: None)
    batch_size = max(batch_size, MIN_PAPERS)
    paper_ids = select_papers(conn, topic, limit=1_000_000)
    _require_min(paper_ids, topic)

    # reparto parejo: n lotes de tamaño casi igual, sin un último lote minúsculo
    n_batches = math.ceil(len(paper_ids) / batch_size)
    base, extra = divmod(len(paper_ids), n_batches)
    batches: list[list[int]] = []
    pos = 0
    for i in range(n_batches):
        size = base + (1 if i < extra else 0)
        batches.append(paper_ids[pos : pos + size])
        pos += size

    sources: list[dict] = []
    partials: list[dict] = []
    for i, batch in enumerate(batches, 1):
        dossiers, batch_sources = build_dossiers(conn, batch, start_n=len(sources) + 1)
        if not batch_sources:
            continue
        sources.extend(batch_sources)
        notify(f"map {i}/{len(batches)}: analizando papers "
               f"[{batch_sources[0]['n']}]–[{batch_sources[-1]['n']}]…")
        data = llm.generate_json(
            SYNTHESIS_PROMPT.format(topic_line=_topic_line(topic), dossiers=dossiers),
            system=SYNTHESIS_SYSTEM,
        )
        partials.append(
            {"first": batch_sources[0]["n"], "last": batch_sources[-1]["n"],
             "sections": _normalize(data)}
        )
    sections = _reduce(partials, topic, notify)
    return _store(conn, topic, sources, sections)


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
    sections = json.loads(row["sections"])
    return Synthesis(
        id=row["id"], topic=row["topic"], sections=sections,
        sources=_sources_for(conn, paper_ids), model=row["model"],
        created_at=row["created_at"],
        audit=audit_citations(sections, len(paper_ids)),
    )


def list_all(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """SELECT id, topic, paper_ids, model, created_at
           FROM syntheses ORDER BY id DESC"""
    ).fetchall()
