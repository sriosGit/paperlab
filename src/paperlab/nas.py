"""Cliente del NAS personal (SRC Cloud) vía su API WebDAV.

Los PDFs se guardan siempre en disco local (ahí los lee `process`) y el NAS
actúa como archivo/respaldo: `sync_pdfs` sube por WebDAV (`/dav/<ruta>`, Basic
auth) los que falten en el NAS y, con `restore`, recupera los que existan en
el NAS pero no en disco (p. ej. tras cambiar de máquina).
"""

import sqlite3
import xml.etree.ElementTree as ET
from collections.abc import Callable
from pathlib import Path
from urllib.parse import unquote

import httpx

from . import config


class NasError(RuntimeError):
    pass


def enabled() -> bool:
    return bool(config.NAS_BASE_URL and config.NAS_USERNAME and config.NAS_PASSWORD)


def _client() -> httpx.Client:
    return httpx.Client(
        base_url=f"{config.NAS_BASE_URL}/dav",
        auth=(config.NAS_USERNAME, config.NAS_PASSWORD),
        timeout=120,
        headers={"User-Agent": config.USER_AGENT},
    )


def _ensure_collections(client: httpx.Client) -> None:
    """MKCOL de cada segmento de NAS_PDF_DIR (201 creado, 405 ya existía)."""
    ruta = ""
    for parte in config.NAS_PDF_DIR.split("/"):
        ruta = f"{ruta}/{parte}"
        resp = client.request("MKCOL", ruta)
        if resp.status_code == 401:
            raise NasError("el NAS rechazó las credenciales (NAS_USERNAME/NAS_PASSWORD)")
        if resp.status_code not in (201, 405):
            raise NasError(f"MKCOL {ruta}: HTTP {resp.status_code}")


def _list_remote(client: httpx.Client) -> set[str]:
    """Nombres de archivo ya presentes en la carpeta del NAS (PROPFIND depth 1)."""
    resp = client.request(
        "PROPFIND", f"/{config.NAS_PDF_DIR}", headers={"Depth": "1"}
    )
    if resp.status_code == 404:
        return set()
    if resp.status_code != 207:
        raise NasError(f"PROPFIND /{config.NAS_PDF_DIR}: HTTP {resp.status_code}")
    nombres = {
        unquote(href.text.rstrip("/").rsplit("/", 1)[-1])
        for href in ET.fromstring(resp.content).iter("{DAV:}href")
        if href.text
    }
    nombres.discard(config.NAS_PDF_DIR.rsplit("/", 1)[-1])  # la colección misma
    return nombres


def sync_pdfs(
    conn: sqlite3.Connection,
    restore: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict:
    """Sube al NAS los PDFs locales que falten allí; con `restore` baja los inversos."""
    notify = progress or (lambda msg: None)
    if not enabled():
        raise NasError("configura NAS_BASE_URL, NAS_USERNAME y NAS_PASSWORD en .env")
    config.ensure_dirs()
    rows = conn.execute("SELECT id, pdf_path FROM papers ORDER BY id").fetchall()
    subidos = en_nas = recuperados = perdidos = 0
    with _client() as client:
        _ensure_collections(client)
        remotos = _list_remote(client)
        for row in rows:
            nombre = f"{row['id']}.pdf"
            local = Path(row["pdf_path"]) if row["pdf_path"] else None
            if local and local.exists():
                if nombre in remotos:
                    en_nas += 1
                    continue
                resp = client.put(f"/{config.NAS_PDF_DIR}/{nombre}", content=local.read_bytes())
                if resp.status_code not in (201, 204):
                    raise NasError(f"PUT {nombre}: HTTP {resp.status_code}")
                subidos += 1
                notify(f"subido {nombre}")
            elif nombre in remotos and restore:
                resp = client.get(f"/{config.NAS_PDF_DIR}/{nombre}")
                if resp.status_code != 200:
                    raise NasError(f"GET {nombre}: HTTP {resp.status_code}")
                destino = config.PDF_DIR / nombre
                destino.write_bytes(resp.content)
                conn.execute(
                    "UPDATE papers SET pdf_path = ? WHERE id = ?", (str(destino), row["id"])
                )
                conn.commit()
                recuperados += 1
                notify(f"recuperado {nombre}")
            elif local:  # la BD apunta a un archivo que no está ni local ni en el NAS
                perdidos += 1
    return {
        "subidos": subidos, "ya_en_nas": en_nas,
        "recuperados": recuperados, "perdidos": perdidos,
    }
