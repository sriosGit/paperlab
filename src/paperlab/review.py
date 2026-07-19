"""Revisión del corpus: excluir papers irrelevantes y consultar su procedencia.

Excluir no borra: el paper se queda en la base (así una búsqueda futura no lo
vuelve a ingerir como nuevo) pero desaparece de resúmenes, RAG, síntesis y
exports. Es reversible con `include`.
"""

import sqlite3


def exclude(conn: sqlite3.Connection, paper_ids: list[int], reason: str = "") -> int:
    if not paper_ids:
        return 0
    conn.executemany(
        "UPDATE papers SET excluded = 1, excluded_reason = ? WHERE id = ?",
        [(reason or None, pid) for pid in paper_ids],
    )
    conn.commit()
    return conn.execute(
        f"SELECT COUNT(*) FROM papers WHERE excluded = 1 AND id IN ({','.join('?' * len(paper_ids))})",
        paper_ids,
    ).fetchone()[0]


def include(conn: sqlite3.Connection, paper_ids: list[int]) -> int:
    if not paper_ids:
        return 0
    conn.executemany(
        "UPDATE papers SET excluded = 0, excluded_reason = NULL WHERE id = ?",
        [(pid,) for pid in paper_ids],
    )
    conn.commit()
    return len(paper_ids)


def provenance(conn: sqlite3.Connection, paper_id: int) -> list[sqlite3.Row]:
    """Búsquedas que trajeron este paper (las más recientes primero)."""
    return conn.execute(
        """SELECT ps.query, ps.source, ps.added_at, s.name AS search_name
           FROM paper_sources ps LEFT JOIN saved_searches s ON s.id = ps.search_id
           WHERE ps.paper_id = ? ORDER BY ps.added_at DESC""",
        (paper_id,),
    ).fetchall()


def by_query(conn: sqlite3.Connection, query: str, only_from_this: bool = False) -> list[sqlite3.Row]:
    """Papers que entraron por una query concreta.

    Con `only_from_this`, solo los que ninguna otra búsqueda trajo también:
    los seguros de descartar si esa consulta resultó ser ruido.
    """
    sql = """SELECT p.* FROM papers p
             JOIN paper_sources ps ON ps.paper_id = p.id
             WHERE ps.query = ?"""
    if only_from_this:
        sql += """ AND NOT EXISTS (
                     SELECT 1 FROM paper_sources o
                     WHERE o.paper_id = p.id AND o.query != ps.query)"""
    return conn.execute(sql + " GROUP BY p.id ORDER BY p.id", (query,)).fetchall()


def queries(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Resumen por query: cuántos papers trajo y cuántos siguen activos."""
    return conn.execute(
        """SELECT ps.query, COUNT(DISTINCT ps.paper_id) AS n_papers,
                  SUM(CASE WHEN p.excluded = 1 THEN 1 ELSE 0 END) AS n_excluidos
           FROM paper_sources ps JOIN papers p ON p.id = ps.paper_id
           GROUP BY ps.query ORDER BY n_papers DESC"""
    ).fetchall()


def stats(conn: sqlite3.Connection) -> dict:
    total = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    excluidos = conn.execute("SELECT COUNT(*) FROM papers WHERE excluded = 1").fetchone()[0]
    sin_procedencia = conn.execute(
        """SELECT COUNT(*) FROM papers p
           WHERE NOT EXISTS (SELECT 1 FROM paper_sources ps WHERE ps.paper_id = p.id)"""
    ).fetchone()[0]
    return {
        "total": total, "activos": total - excluidos,
        "excluidos": excluidos, "sin_procedencia": sin_procedencia,
    }
