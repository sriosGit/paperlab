"""Unpaywall: dado un DOI, localiza el PDF open-access. https://unpaywall.org/products/api"""

import httpx

from .. import config

API_URL = "https://api.unpaywall.org/v2"


def find_pdf_url(doi: str) -> str | None:
    if not config.CONTACT_EMAIL:
        return None  # Unpaywall exige email
    try:
        resp = httpx.get(
            f"{API_URL}/{doi}",
            params={"email": config.CONTACT_EMAIL},
            timeout=30,
            headers={"User-Agent": config.USER_AGENT},
        )
        if resp.status_code != 200:
            return None
        best = resp.json().get("best_oa_location") or {}
        return best.get("url_for_pdf")
    except httpx.HTTPError:
        return None
