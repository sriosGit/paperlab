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


def _get_work(client: httpx.Client, ref: str) -> dict | None:
    params = {"mailto": config.CONTACT_EMAIL} if config.CONTACT_EMAIL else {}
    resp = client.get(f"{API_URL}/{ref}", params=params)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def fetch_by_ids(doi: str | None, arxiv_id: str | None) -> Paper | None:
    """Busca un work concreto por DOI o arXiv id (para enriquecer papers locales).

    OpenAlex indexa los preprints de arXiv con DOI `10.48550/arXiv.<id>`, así que
    ese es el camino cuando solo hay arxiv_id.
    """
    with httpx.Client(timeout=60, headers={"User-Agent": config.USER_AGENT}) as client:
        if doi:
            w = _get_work(client, f"doi:{doi}")
            if w:
                return _work_to_paper(w)
        if arxiv_id:
            base = arxiv_id.split("v")[0]  # quita el sufijo de versión (v1, v2…)
            w = _get_work(client, f"doi:10.48550/arXiv.{base}")
            if w:
                return _work_to_paper(w)
    return None


def fetch_by_openalex_ids(ids: list[str]) -> list[Paper]:
    """Trae varios works por su OpenAlex id (para snowballing de citas).

    En lotes de 50 vía el filtro `openalex_id:A|B|C` — más allá de eso la URL
    se vuelve poco fiable en algunos proxies/servidores.
    """
    papers: list[Paper] = []
    with httpx.Client(timeout=60, headers={"User-Agent": config.USER_AGENT}) as client:
        for i in range(0, len(ids), 50):
            batch = ids[i : i + 50]
            params: dict = {"filter": "openalex_id:" + "|".join(batch), "per-page": len(batch)}
            if config.CONTACT_EMAIL:
                params["mailto"] = config.CONTACT_EMAIL
            resp = client.get(API_URL, params=params)
            resp.raise_for_status()
            for w in resp.json().get("results", []):
                papers.append(_work_to_paper(w))
    return papers


def search(
    query: str, limit: int, from_year: int | None = None, to_year: int | None = None
) -> list[Paper]:
    papers: list[Paper] = []
    params: dict = {"search": query, "per-page": min(PAGE_SIZE, limit)}
    filters = []
    if from_year:
        filters.append(f"from_publication_date:{from_year}-01-01")
    if to_year:
        filters.append(f"to_publication_date:{to_year}-12-31")
    if filters:
        params["filter"] = ",".join(filters)
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
