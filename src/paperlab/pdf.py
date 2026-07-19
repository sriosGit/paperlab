"""Descarga de PDFs y extracción de texto."""

import sqlite3
from collections.abc import Iterator

import fitz  # PyMuPDF
import httpx

from . import config
from .ingest import semanticscholar, unpaywall


def _candidate_urls(row: sqlite3.Row) -> Iterator[str]:
    """URLs candidatas en orden de preferencia.

    Generador a propósito: las APIs (Unpaywall, Semantic Scholar) solo se
    consultan si las URLs anteriores fallaron.
    """
    if row["pdf_url"]:
        yield row["pdf_url"]
    if row["arxiv_id"]:
        yield f"https://arxiv.org/pdf/{row['arxiv_id']}"
    if row["doi"]:
        url = unpaywall.find_pdf_url(row["doi"])
        if url:
            yield url
        url = semanticscholar.find_pdf_url(row["doi"])
        if url:
            yield url


def _download(client: httpx.Client, url: str) -> bytes | None:
    """Devuelve el contenido si la URL responde con un PDF real; si no, None."""
    try:
        resp = client.get(url)
        resp.raise_for_status()
    except httpx.HTTPError:
        return None
    if not resp.content.startswith(b"%PDF"):
        return None  # landing page HTML, paywall, etc.
    return resp.content


def fetch_pdfs(conn: sqlite3.Connection, limit: int | None = None, retry: bool = False) -> dict:
    """Descarga PDFs pendientes probando cada URL candidata hasta dar con un PDF real.

    Por defecto solo papers en estado 'new'; con `retry` cualquier paper sin PDF
    descargado, sea cual sea su estado. Si un paper ya indexado consigue PDF, sus
    chunks (que eran solo del abstract) se borran para que `process` re-indexe
    el texto completo.
    """
    config.ensure_dirs()
    where = "pdf_path IS NULL" if retry else "status = 'new'"
    rows = conn.execute(
        f"SELECT * FROM papers WHERE {where} ORDER BY id"
        + (f" LIMIT {int(limit)}" if limit else "")
    ).fetchall()
    ok = failed = no_url = 0
    with httpx.Client(
        timeout=90, follow_redirects=True, headers={"User-Agent": config.USER_AGENT}
    ) as client:
        for row in rows:
            content = None
            tried: set[str] = set()
            for url in _candidate_urls(row):
                if url in tried:
                    continue
                tried.add(url)
                content = _download(client, url)
                if content:
                    break
            if not tried:
                no_url += 1
                continue
            if content is None:
                failed += 1
                continue
            path = config.PDF_DIR / f"{row['id']}.pdf"
            path.write_bytes(content)
            if row["status"] == "new":
                conn.execute(
                    "UPDATE papers SET pdf_path = ?, status = 'fetched' WHERE id = ?",
                    (str(path), row["id"]),
                )
            else:
                conn.execute("DELETE FROM chunks WHERE paper_id = ?", (row["id"],))
                conn.execute(
                    "UPDATE papers SET pdf_path = ? WHERE id = ?", (str(path), row["id"])
                )
            conn.commit()
            ok += 1
    return {"descargados": ok, "fallidos": failed, "sin_url": no_url, "pendientes": len(rows)}


def extract_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    try:
        pages = [page.get_text("text") for page in doc]
    finally:
        doc.close()
    text = "\n".join(pages)
    # limpieza mínima: colapsar saltos de línea múltiples
    lines = [ln.strip() for ln in text.splitlines()]
    out: list[str] = []
    blank = False
    for ln in lines:
        if ln:
            out.append(ln)
            blank = False
        elif not blank:
            out.append("")
            blank = True
    return "\n".join(out)


def chunk_text(text: str, title: str, chunk_chars: int = 3200, overlap: int = 300) -> list[str]:
    """Trocea por párrafos hasta ~chunk_chars (≈800 tokens), con solapamiento."""
    header = f"[{title}]\n"
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) + 2 > chunk_chars and current:
            chunks.append(header + current.strip())
            current = current[-overlap:] if overlap else ""
        # párrafos enormes (p. ej. sin saltos): trocear duro
        while len(para) > chunk_chars:
            chunks.append(header + (current + "\n\n" + para[:chunk_chars]).strip())
            para = para[chunk_chars - overlap:]
            current = ""
        current = (current + "\n\n" + para).strip()
    if current.strip():
        chunks.append(header + current.strip())
    return chunks
