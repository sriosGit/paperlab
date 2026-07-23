"""Fetcher de arXiv (API Atom, sin key). https://info.arxiv.org/help/api/"""

import time

import feedparser
import httpx

from .. import config
from ..models import Paper

API_URL = "https://export.arxiv.org/api/query"
PAGE_SIZE = 100  # arXiv pide no pasar de ~100 por petición y esperar 3 s entre páginas


def _entry_to_paper(entry) -> Paper:
    # entry.id: http://arxiv.org/abs/2401.12345v2 → 2401.12345
    arxiv_id = entry.id.rsplit("/abs/", 1)[-1]
    if "v" in arxiv_id.rsplit("/", 1)[-1]:
        base, _, ver = arxiv_id.rpartition("v")
        if ver.isdigit():
            arxiv_id = base
    doi = getattr(entry, "arxiv_doi", None)
    year = None
    if getattr(entry, "published_parsed", None):
        year = entry.published_parsed.tm_year
    category = None
    if getattr(entry, "arxiv_primary_category", None):
        category = entry.arxiv_primary_category.get("term")
    return Paper(
        arxiv_id=arxiv_id,
        doi=doi,
        title=" ".join(entry.title.split()),
        abstract=" ".join(entry.summary.split()) if getattr(entry, "summary", None) else None,
        authors=[a.name for a in getattr(entry, "authors", [])],
        year=year,
        venue=category,
        source="arxiv",
        url=f"https://arxiv.org/abs/{arxiv_id}",
        pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
    )


def search(
    query: str, limit: int, from_year: int | None = None, to_year: int | None = None
) -> list[Paper]:
    search_query = f"all:{query}"
    if from_year or to_year:
        start_date = f"{from_year or 1990}01010000"
        end_date = f"{to_year or 2100}12312359"
        search_query += f" AND submittedDate:[{start_date} TO {end_date}]"
    papers: list[Paper] = []
    with httpx.Client(timeout=60, headers={"User-Agent": config.USER_AGENT}) as client:
        start = 0
        while start < limit:
            batch = min(PAGE_SIZE, limit - start)
            resp = client.get(
                API_URL,
                params={
                    "search_query": search_query,
                    "start": start,
                    "max_results": batch,
                    "sortBy": "relevance",
                },
            )
            resp.raise_for_status()
            feed = feedparser.parse(resp.text)
            if not feed.entries:
                break
            papers.extend(_entry_to_paper(e) for e in feed.entries)
            start += len(feed.entries)
            if start < limit:
                time.sleep(3)
    return papers
