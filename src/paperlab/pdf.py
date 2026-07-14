"""Descarga de PDFs y extracción de texto."""

import sqlite3

import fitz  # PyMuPDF
import httpx

from . import config
from .ingest import unpaywall


def _resolve_pdf_url(row: sqlite3.Row) -> str | None:
    if row["pdf_url"]:
        return row["pdf_url"]
    if row["arxiv_id"]:
        return f"https://arxiv.org/pdf/{row['arxiv_id']}"
    if row["doi"]:
        return unpaywall.find_pdf_url(row["doi"])
    return None


def fetch_pdfs(conn: sqlite3.Connection, limit: int | None = None) -> dict:
    """Descarga PDFs de los papers en estado 'new'. Devuelve conteos."""
    config.ensure_dirs()
    rows = conn.execute(
        "SELECT * FROM papers WHERE status = 'new' ORDER BY id"
        + (f" LIMIT {int(limit)}" if limit else "")
    ).fetchall()
    ok = failed = no_url = 0
    with httpx.Client(
        timeout=90, follow_redirects=True, headers={"User-Agent": config.USER_AGENT}
    ) as client:
        for row in rows:
            url = _resolve_pdf_url(row)
            if not url:
                no_url += 1
                continue
            path = config.PDF_DIR / f"{row['id']}.pdf"
            try:
                resp = client.get(url)
                resp.raise_for_status()
                if not resp.content.startswith(b"%PDF"):
                    raise ValueError("la respuesta no es un PDF")
                path.write_bytes(resp.content)
            except (httpx.HTTPError, ValueError):
                failed += 1
                continue
            conn.execute(
                "UPDATE papers SET pdf_path = ?, status = 'fetched' WHERE id = ?",
                (str(path), row["id"]),
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
