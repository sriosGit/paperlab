"""Tests de exclusión de papers y procedencia."""

import sqlite3

import pytest

from paperlab import analyze, db, review, synthesize
from paperlab.ingest.base import store_papers
from paperlab.models import Paper


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript(db.SCHEMA)
    yield c
    c.close()


def _add_paper(conn, id, title="T", abstract="abs"):
    conn.execute(
        """INSERT INTO papers (id, title, title_norm, source, abstract, authors)
           VALUES (?, ?, ?, 'arxiv', ?, '[]')""",
        (id, title, title.lower(), abstract),
    )


def _add_summary(conn, paper_id):
    conn.execute(
        "INSERT INTO summaries (paper_id, summary_md, findings, model) VALUES (?, 'R', '[]', 'x')",
        (paper_id,),
    )


# --- exclusión ---

def test_exclude_e_include_con_motivo(conn):
    _add_paper(conn, 1)
    assert review.exclude(conn, [1], "fuera de tema") == 1
    row = conn.execute("SELECT * FROM papers WHERE id = 1").fetchone()
    assert row["excluded"] == 1 and row["excluded_reason"] == "fuera de tema"
    review.include(conn, [1])
    row = conn.execute("SELECT * FROM papers WHERE id = 1").fetchone()
    assert row["excluded"] == 0 and row["excluded_reason"] is None


def test_excluido_sale_de_resumenes_pendientes_y_chunks(conn):
    _add_paper(conn, 1)
    _add_paper(conn, 2)
    review.exclude(conn, [2])
    assert analyze.pending_summaries(conn) == [1]
    analyze.ensure_chunks(conn)
    troceados = {r["paper_id"] for r in conn.execute("SELECT DISTINCT paper_id FROM chunks")}
    assert troceados == {1}


def test_excluido_sale_de_seleccion_para_sintesis(conn):
    for i in (1, 2, 3):
        _add_paper(conn, i, title=f"Grafeno {i}")
        _add_summary(conn, i)
    review.exclude(conn, [2])
    assert synthesize.select_papers(conn, None, 10) == [3, 1]
    assert 2 not in synthesize.select_papers(conn, "grafeno", 10)


def test_excluido_sale_del_rag(conn, monkeypatch):
    _add_paper(conn, 1, title="Uno", abstract="grafeno conductor")
    _add_paper(conn, 2, title="Dos", abstract="grafeno conductor")
    analyze.ensure_chunks(conn)
    review.exclude(conn, [2])
    monkeypatch.setattr(analyze.llm, "embed", lambda t: (_ for _ in ()).throw(analyze.llm.OllamaError("sin ollama")))
    chunks = analyze.hybrid_search(conn, "grafeno", k=10)
    assert {c["paper_id"] for c in chunks} == {1}


def test_excluido_no_se_reingiere_como_nuevo(conn):
    store_papers(conn, [Paper(doi="10.1/x", title="Ruido", source="openalex")], query="ia")
    pid = conn.execute("SELECT id FROM papers").fetchone()["id"]
    review.exclude(conn, [pid], "ruido")
    ins, dup = store_papers(conn, [Paper(doi="10.1/x", title="Ruido", source="openalex")], query="ia")
    assert (ins, dup) == (0, 1)
    assert conn.execute("SELECT excluded FROM papers WHERE id = ?", (pid,)).fetchone()[0] == 1


# --- procedencia ---

def test_store_papers_registra_procedencia(conn):
    p = Paper(doi="10.1/x", title="Uno", source="openalex")
    store_papers(conn, [p], query="inteligencia artificial", search_id=None)
    pid = conn.execute("SELECT id FROM papers").fetchone()["id"]
    prov = review.provenance(conn, pid)
    assert len(prov) == 1
    assert prov[0]["query"] == "inteligencia artificial" and prov[0]["source"] == "openalex"


def test_procedencia_acumula_varias_busquedas(conn):
    p = Paper(doi="10.1/x", title="Uno", source="openalex")
    store_papers(conn, [p], query="ia")
    store_papers(conn, [p], query="agentes")
    store_papers(conn, [p], query="ia")  # repetida: no duplica
    pid = conn.execute("SELECT id FROM papers").fetchone()["id"]
    assert {r["query"] for r in review.provenance(conn, pid)} == {"ia", "agentes"}
    assert len(review.provenance(conn, pid)) == 2


def test_by_query_only_aisla_los_exclusivos(conn):
    a = Paper(doi="10.1/a", title="Solo ia", source="openalex")
    b = Paper(doi="10.1/b", title="Ambas", source="openalex")
    store_papers(conn, [a, b], query="ia")
    store_papers(conn, [b], query="agentes")
    todos = {r["title"] for r in review.by_query(conn, "ia")}
    solo = {r["title"] for r in review.by_query(conn, "ia", only_from_this=True)}
    assert todos == {"Solo ia", "Ambas"}
    assert solo == {"Solo ia"}


def test_stats_y_queries(conn):
    store_papers(conn, [Paper(doi="10.1/a", title="A", source="openalex")], query="ia")
    store_papers(conn, [Paper(doi="10.1/b", title="B", source="openalex")], query="ia")
    _add_paper(conn, 99, title="Sin procedencia")
    pid = conn.execute("SELECT id FROM papers WHERE title = 'A'").fetchone()["id"]
    review.exclude(conn, [pid])
    s = review.stats(conn)
    assert s == {"total": 3, "active": 2, "excluded": 1, "without_provenance": 1}
    q = review.queries(conn)[0]
    assert q["query"] == "ia" and q["n_papers"] == 2 and q["n_excluded"] == 1


# --- guarda de modelo de embeddings ---

def test_ensure_embeddings_recalcula_al_cambiar_de_modelo(conn, monkeypatch):
    from paperlab import config
    _add_paper(conn, 1)
    conn.execute("INSERT INTO chunks (paper_id, seq, text) VALUES (1, 0, 'texto')")
    monkeypatch.setattr(config, "OLLAMA_EMBED_MODEL", "nomic-embed-text")
    monkeypatch.setattr(analyze.llm, "embed", lambda ts: [[0.1, 0.2] for _ in ts])
    assert analyze.ensure_embeddings(conn) == 1
    assert analyze.stale_embeddings(conn) == 0
    assert analyze.ensure_embeddings(conn) == 0          # ya está, no repite

    monkeypatch.setattr(config, "OLLAMA_EMBED_MODEL", "bge-m3")
    assert analyze.stale_embeddings(conn) == 1           # detecta el obsoleto
    monkeypatch.setattr(analyze.llm, "embed", lambda ts: [[0.1, 0.2, 0.3, 0.4] for _ in ts])
    assert analyze.ensure_embeddings(conn) == 1          # lo recalcula
    assert analyze.stale_embeddings(conn) == 0
    row = conn.execute("SELECT embed_model FROM chunks WHERE id = 1").fetchone()
    assert row["embed_model"] == "bge-m3"


def test_indice_vectorial_ignora_vectores_de_otro_modelo(conn, monkeypatch):
    """Mezclar dimensiones rompería la matriz numpy: deben filtrarse."""
    from paperlab import config
    _add_paper(conn, 1)
    conn.execute("INSERT INTO chunks (paper_id, seq, text) VALUES (1, 0, 'viejo')")
    conn.execute("INSERT INTO chunks (paper_id, seq, text) VALUES (1, 1, 'nuevo')")
    conn.execute(
        "UPDATE chunks SET embedding = ?, embed_model = 'nomic-embed-text' WHERE seq = 0",
        (db.embedding_to_blob([0.1, 0.2]),),
    )
    conn.execute(
        "UPDATE chunks SET embedding = ?, embed_model = 'bge-m3' WHERE seq = 1",
        (db.embedding_to_blob([0.1, 0.2, 0.3, 0.4]),),
    )
    conn.commit()
    monkeypatch.setattr(config, "OLLAMA_EMBED_MODEL", "bge-m3")
    idx = analyze._VectorIndex()
    ids = idx.search(conn, [0.1, 0.2, 0.3, 0.4], k=10)
    assert idx.matrix.shape == (1, 4)     # solo el vector de bge-m3
    assert ids == [conn.execute("SELECT id FROM chunks WHERE seq = 1").fetchone()["id"]]
