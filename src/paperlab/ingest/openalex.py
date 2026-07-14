"""Fetcher de OpenAlex (sin key; email para el polite pool). https://docs.openalex.org/"""

import httpx

from .. import config
from ..models import Paper, normalize_doi

API_URL = "https://api.openalex.org/works"
PAGE_SIZE = 200


def _reconstruct_abstract(inverted: dict | None) -> str | None:
    """OpenAlex entrega el abstract como índice invertido {palabra: [posiciones]}."""
    if not inverted:
        return None
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted.items():
        positions.extend((i, word) for i in idxs)
    positions.sort()
    return " ".join(word for _, word in positions)


def _work_to_paper(w: dict) -> Paper:
    openalex_id = w["id"].rsplit("/", 1)[-1]
    arxiv_id = None
    for loc in w.get("locations") or []:
        landing = (loc.get("landing_page_url") or "")
        if "arxiv.org/abs/" in landing:
            arxiv_id = landing.rsplit("/abs/", 1)[-1]
            break
    primary = w.get("primary_location") or {}
    venue = (primary.get("source") or {}).get("display_name")
    best_oa = w.get("best_oa_location") or {}
    return Paper(
        doi=normalize_doi(w.get("doi")),
        arxiv_id=arxiv_id,
        openalex_id=openalex_id,
        title=w.get("display_name") or "(sin título)",
        abstract=_reconstruct_abstract(w.get("abstract_inverted_index")),
        authors=[
            (a.get("author") or {}).get("display_name", "")
            for a in w.get("authorships") or []
        ],
        year=w.get("publication_year"),
        venue=venue,
        source="openalex",
        url=primary.get("landing_page_url") or w.get("id"),
        pdf_url=best_oa.get("pdf_url"),
        referenced_ids=[r.rsplit("/", 1)[-1] for r in w.get("referenced_works") or []],
    )


def search(query: str, limit: int) -> list[Paper]:
    papers: list[Paper] = []
    params: dict = {"search": query, "per-page": min(PAGE_SIZE, limit)}
    if config.CONTACT_EMAIL:
        params["mailto"] = config.CONTACT_EMAIL
    cursor = "*"
    with httpx.Client(timeout=60, headers={"User-Agent": config.USER_AGENT}) as client:
        while len(papers) < limit and cursor:
            resp = client.get(API_URL, params={**params, "cursor": cursor})
            resp.raise_for_status()
            data = resp.json()
            for w in data.get("results", []):
                papers.append(_work_to_paper(w))
                if len(papers) >= limit:
                    break
            cursor = (data.get("meta") or {}).get("next_cursor")
    return papers
