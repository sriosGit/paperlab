"""Traduce una petición en lenguaje natural a una query optimizada para las fuentes.

arXiv y OpenAlex hacen coincidencia de texto simple, no entienden lenguaje
natural: una pregunta como "papers sobre RAG en los últimos 5 años" hay que
convertirla en palabras clave en inglés más, aparte, un filtro de fecha nativo
de cada API (ver `ingest.arxiv`/`ingest.openalex`).
"""

from dataclasses import dataclass, field
from datetime import datetime

from . import llm

VALID_SOURCES = {"arxiv", "openalex"}

SYSTEM = (
    "Eres un experto en búsqueda bibliográfica académica. Traduces peticiones en "
    "lenguaje natural (español o inglés) a queries optimizadas para las APIs de "
    "arXiv y OpenAlex, que hacen coincidencia de texto simple (no entienden lenguaje "
    "natural ni booleanos complejos). Usa palabras clave en inglés, las más "
    "distintivas del tema, sin relleno ni frases completas."
)

PROMPT = """Petición del usuario: "{request}"

Año actual: {current_year}.

Devuelve SOLO un objeto JSON con estas claves:
- "query": 3-8 palabras clave en inglés que maximicen precisión y recall en arXiv/OpenAlex (sin comillas ni operadores, solo términos separados por espacios)
- "sources": lista con las fuentes relevantes, subconjunto de ["arxiv", "openalex"] (usa ambas salvo que la petición pida explícitamente una sola)
- "from_year": año inicial del rango si la petición menciona un periodo (entero o null)
- "to_year": año final del rango (entero o null; si dice "últimos N años" usa {current_year} como to_year y {current_year}-N como from_year)
- "notes": una frase breve en español explicando la interpretación (qué palabras elegiste y por qué)
"""


@dataclass
class QueryPlan:
    query: str
    sources: list[str] = field(default_factory=lambda: ["arxiv", "openalex"])
    from_year: int | None = None
    to_year: int | None = None
    notes: str = ""


def _to_year(v) -> int | None:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def suggest(request: str) -> QueryPlan:
    """Pide al LLM local una query + filtros a partir de una petición en lenguaje natural."""
    current_year = datetime.now().year
    data = llm.generate_json(
        PROMPT.format(request=request.strip(), current_year=current_year), system=SYSTEM
    )
    query = str(data.get("query") or "").strip()
    if not query:
        raise llm.OllamaError(f"el modelo no devolvió una query utilizable: {data}")
    sources = [s for s in data.get("sources") or [] if s in VALID_SOURCES] or ["arxiv", "openalex"]
    return QueryPlan(
        query=query,
        sources=sources,
        from_year=_to_year(data.get("from_year")),
        to_year=_to_year(data.get("to_year")),
        notes=str(data.get("notes") or "").strip(),
    )
