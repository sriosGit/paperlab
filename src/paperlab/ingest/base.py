"""Alta en base de datos con deduplicación, común a todas las fuentes."""

import json
import sqlite3

from ..models import Paper, normalize_title


def _find_existing(conn: sqlite3.Connection, p: Paper) -> sqlite3.Row | None:
    for col, val in (("doi", p.doi), ("arxiv_id", p.arxiv_id), ("openalex_id", p.openalex_id)):
        if val:
            row = conn.execute(f"SELECT * FROM papers WHERE {col} = ?", (val,)).fetchone()
            if row:
                return row
    return conn.execute(
        "SELECT * FROM papers WHERE title_norm = ?", (normalize_title(p.title),)
    ).fetchone()


def store_papers(
    conn: sqlite3.Connection,
    papers: list[Paper],
    query: str | None = None,
    search_id: int | None = None,
) -> tuple[int, int]:
    """Inserta papers nuevos; a los duplicados les completa IDs/PDF que falten.

    Con `query` deja rastro de procedencia en `paper_sources` (también para los
    duplicados: un mismo paper puede llegar por varias búsquedas).

    Devuelve (insertados, duplicados).
    """
    inserted = duplicates = 0
    for p in papers:
        existing = _find_existing(conn, p)
        if existing:
            duplicates += 1
            updates = {}
            for col in ("doi", "arxiv_id", "openalex_id", "pdf_url", "abstract"):
                if getattr(p, col, None) and not existing[col]:
                    updates[col] = getattr(p, col)
            if updates:
                sets = ", ".join(f"{c} = ?" for c in updates)
                conn.execute(
                    f"UPDATE papers SET {sets} WHERE id = ?",
                    (*updates.values(), existing["id"]),
                )
            paper_id = existing["id"]
        else:
            cur = conn.execute(
                """INSERT INTO papers (doi, arxiv_id, openalex_id, title, title_norm,
                       abstract, authors, year, venue, source, url, pdf_url)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    p.doi, p.arxiv_id, p.openalex_id, p.title, normalize_title(p.title),
                    p.abstract, json.dumps(p.authors, ensure_ascii=False), p.year,
                    p.venue, p.source, p.url, p.pdf_url,
                ),
            )
            paper_id = cur.lastrowid
            inserted += 1
        if query:
            conn.execute(
                """INSERT OR IGNORE INTO paper_sources (paper_id, search_id, query, source)
                   VALUES (?, ?, ?, ?)""",
                (paper_id, search_id, query, p.source),
            )
        for cited in p.referenced_ids:
            conn.execute(
                "INSERT OR IGNORE INTO citations (paper_id, cited_openalex_id) VALUES (?, ?)",
                (paper_id, cited),
            )
    conn.commit()
    return inserted, duplicates


def run_search(
    conn: sqlite3.Connection,
    query: str,
    sources: list[str],
    limit: int,
    search_id: int | None = None,
    from_year: int | None = None,
    to_year: int | None = None,
) -> dict:
    """Ejecuta la búsqueda en cada fuente y guarda resultados. Devuelve conteos."""
    from . import FETCHERS

    result = {}
    for source in sources:
        fetch = FETCHERS.get(source.strip())
        if not fetch:
            result[source] = {"error": f"fuente desconocida: {source}"}
            continue
        try:
            papers = fetch(query, limit, from_year=from_year, to_year=to_year)
        except Exception as e:  # noqa: BLE001 — la fuente no debe tumbar el resto
            result[source] = {"error": str(e)}
            continue
        ins, dup = store_papers(conn, papers, query=query, search_id=search_id)
        result[source] = {"encontrados": len(papers), "nuevos": ins, "duplicados": dup}
    return result
