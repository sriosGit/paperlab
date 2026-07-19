"""Semantic Scholar Graph API: localiza el PDF open-access por DOI (sin API key).

Buena cobertura de repositorios iberoamericanos (SciELO, Redalyc) donde
Unpaywall a veces no tiene el enlace. https://api.semanticscholar.org/
"""

import httpx

from .. import config

API_URL = "https://api.semanticscholar.org/graph/v1/paper"


def find_pdf_url(doi: str) -> str | None:
    try:
        resp = httpx.get(
            f"{API_URL}/DOI:{doi}",
            params={"fields": "openAccessPdf"},
            timeout=30,
            headers={"User-Agent": config.USER_AGENT},
        )
        if resp.status_code != 200:  # 404 = no indexado; 429 = rate limit compartido
            return None
        return (resp.json().get("openAccessPdf") or {}).get("url")
    except httpx.HTTPError:
        return None
