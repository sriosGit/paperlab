"""Modelos normalizados comunes a todas las fuentes."""

from pydantic import BaseModel


class Paper(BaseModel):
    doi: str | None = None
    arxiv_id: str | None = None
    openalex_id: str | None = None
    title: str
    abstract: str | None = None
    authors: list[str] = []
    year: int | None = None
    venue: str | None = None
    source: str  # arxiv | openalex | pubmed
    url: str | None = None
    pdf_url: str | None = None
    referenced_ids: list[str] = []  # OpenAlex IDs de trabajos citados


def normalize_doi(doi: str | None) -> str | None:
    if not doi:
        return None
    doi = doi.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
    return doi or None


def normalize_title(title: str) -> str:
    """Clave de deduplicación por título: minúsculas y solo alfanuméricos."""
    return "".join(c for c in title.lower() if c.isalnum())
