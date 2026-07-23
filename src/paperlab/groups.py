"""Grupos: un espacio persistente que junta búsquedas, papers y sus síntesis.

Una búsqueda guardada trae papers y cada paper puede tener resumen (hallazgos/
método/relevancia en `summaries`), pero nada los junta bajo un mismo proyecto
de investigación. Un grupo liga búsquedas guardadas (`group_searches`), papers
sueltos añadidos a mano o por citación (`group_papers`, con `added_via`) y
sirve de ancla para sintetizarlos juntos (`synthesize.run*(group_id=...)`).

`expand_citations` es la "búsqueda profunda": resuelve contra OpenAlex los
`cited_openalex_id` de `citations` que los papers del grupo ya tienen
registrados pero que no están en la biblioteca, los ingiere y los suma al
grupo. Como `ingest.base.store_papers` vuelca a su vez las citas de esos
papers nuevos, repetir la acción profundiza otro nivel.
"""

import sqlite3

from .ingest import openalex
from .ingest.base import store_papers


def create(conn: sqlite3.Connection, name: str, description: str = "") -> int:
    cur = conn.execute(
        "INSERT INTO groups (name, description) VALUES (?, ?)",
        (name, description or None),
    )
    conn.commit()
    return cur.lastrowid


def list_all(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """SELECT g.*,
                  (SELECT COUNT(*) FROM group_papers gp WHERE gp.group_id = g.id) AS n_papers,
                  (SELECT COUNT(*) FROM group_searches gs WHERE gs.group_id = g.id) AS n_searches
           FROM groups g ORDER BY g.id DESC"""
    ).fetchall()


def get(conn: sqlite3.Connection, group_id: int) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM groups WHERE id = ?", (group_id,)).fetchone()


def delete(conn: sqlite3.Connection, group_id: int) -> None:
    conn.execute("DELETE FROM groups WHERE id = ?", (group_id,))
    conn.commit()


def link_search(conn: sqlite3.Connection, group_id: int, search_id: int) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO group_searches (group_id, search_id) VALUES (?, ?)",
        (group_id, search_id),
    )
    conn.commit()
    sync_search(conn, search_id)


def unlink_search(conn: sqlite3.Connection, group_id: int, search_id: int) -> None:
    conn.execute(
        "DELETE FROM group_searches WHERE group_id = ? AND search_id = ?",
        (group_id, search_id),
    )
    conn.commit()


def linked_searches(conn: sqlite3.Connection, group_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        """SELECT s.* FROM saved_searches s
           JOIN group_searches gs ON gs.search_id = s.id
           WHERE gs.group_id = ? ORDER BY s.id""",
        (group_id,),
    ).fetchall()


def sync_search(conn: sqlite3.Connection, search_id: int) -> int:
    """Vuelca a `group_papers` los papers que esa búsqueda ya trajo, para todo grupo que la tenga ligada.

    Se llama tras ejecutar una búsqueda guardada (recién ligada o reejecutada).
    """
    cur = conn.execute(
        """INSERT OR IGNORE INTO group_papers (group_id, paper_id, added_via)
           SELECT gs.group_id, ps.paper_id, 'search'
           FROM group_searches gs
           JOIN paper_sources ps ON ps.search_id = gs.search_id
           WHERE gs.search_id = ?""",
        (search_id,),
    )
    conn.commit()
    return cur.rowcount


def add_paper(
    conn: sqlite3.Connection, group_id: int, paper_id: int, added_via: str = "manual"
) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO group_papers (group_id, paper_id, added_via) VALUES (?, ?, ?)",
        (group_id, paper_id, added_via),
    )
    conn.commit()


def remove_paper(conn: sqlite3.Connection, group_id: int, paper_id: int) -> None:
    conn.execute(
        "DELETE FROM group_papers WHERE group_id = ? AND paper_id = ?",
        (group_id, paper_id),
    )
    conn.commit()


def members(conn: sqlite3.Connection, group_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        """SELECT p.id, p.title, p.year, p.source, p.status, p.excluded,
                  gp.added_via, gp.added_at,
                  s.findings, s.method, s.relevance
           FROM group_papers gp
           JOIN papers p ON p.id = gp.paper_id
           LEFT JOIN summaries s ON s.paper_id = p.id
           WHERE gp.group_id = ?
           ORDER BY gp.added_at DESC""",
        (group_id,),
    ).fetchall()


def pending_citations(conn: sqlite3.Connection, group_id: int, limit: int = 200) -> list[str]:
    """OpenAlex ids citados por papers del grupo que aún no están en la biblioteca."""
    rows = conn.execute(
        """SELECT DISTINCT c.cited_openalex_id AS id
           FROM citations c
           JOIN group_papers gp ON gp.paper_id = c.paper_id
           WHERE gp.group_id = ?
             AND NOT EXISTS (SELECT 1 FROM papers p WHERE p.openalex_id = c.cited_openalex_id)
           LIMIT ?""",
        (group_id, limit),
    ).fetchall()
    return [r["id"] for r in rows]


def expand_citations(conn: sqlite3.Connection, group_id: int, limit: int = 200) -> dict:
    """Trae al corpus los papers citados por los del grupo y los suma al grupo."""
    ids = pending_citations(conn, group_id, limit)
    if not ids:
        return {"pending": 0, "fetched": 0, "inserted": 0, "duplicates": 0}
    papers = openalex.fetch_by_openalex_ids(ids)
    inserted, duplicates = store_papers(conn, papers)
    if papers:
        openalex_ids = [p.openalex_id for p in papers if p.openalex_id]
        placeholders = ",".join("?" * len(openalex_ids))
        rows = conn.execute(
            f"SELECT id FROM papers WHERE openalex_id IN ({placeholders})", openalex_ids
        ).fetchall()
        for row in rows:
            add_paper(conn, group_id, row["id"], added_via="citation")
    return {
        "pending": len(ids),
        "fetched": len(papers),
        "inserted": inserted,
        "duplicates": duplicates,
    }
