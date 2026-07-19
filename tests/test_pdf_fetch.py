"""Tests de la cadena de fuentes de PDF y del modo retry (sin red)."""

import sqlite3

import pytest

from paperlab import config, db
from paperlab import pdf as pdf_mod


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript(db.SCHEMA)
    yield c
    c.close()


def _add_paper(conn, *, id, status="new", pdf_url=None, doi=None, arxiv_id=None):
    conn.execute(
        """INSERT INTO papers (id, title, title_norm, source, status, pdf_url, doi, arxiv_id, authors)
           VALUES (?, 'T', 't', 'arxiv', ?, ?, ?, ?, '[]')""",
        (id, status, pdf_url, doi, arxiv_id),
    )


def _row(conn, id):
    return conn.execute("SELECT * FROM papers WHERE id = ?", (id,)).fetchone()


# --- cadena de candidatos ---

def test_candidatos_en_orden_y_lazy(conn, monkeypatch):
    llamadas = []
    monkeypatch.setattr(pdf_mod.unpaywall, "find_pdf_url",
                        lambda doi: llamadas.append("unpaywall") or "http://up/x.pdf")
    monkeypatch.setattr(pdf_mod.semanticscholar, "find_pdf_url",
                        lambda doi: llamadas.append("s2") or "http://s2/x.pdf")
    _add_paper(conn, id=1, pdf_url="http://oa/x.pdf", arxiv_id="2401.1", doi="10.1/x")
    gen = pdf_mod._candidate_urls(_row(conn, 1))
    assert next(gen) == "http://oa/x.pdf"
    assert next(gen) == "https://arxiv.org/pdf/2401.1"
    assert llamadas == []  # las APIs no se tocan hasta necesitarlas
    assert next(gen) == "http://up/x.pdf"
    assert next(gen) == "http://s2/x.pdf"
    assert llamadas == ["unpaywall", "s2"]


def test_candidatos_omite_apis_sin_respuesta(conn, monkeypatch):
    monkeypatch.setattr(pdf_mod.unpaywall, "find_pdf_url", lambda doi: None)
    monkeypatch.setattr(pdf_mod.semanticscholar, "find_pdf_url", lambda doi: "http://s2/x.pdf")
    _add_paper(conn, id=1, doi="10.1/x")
    assert list(pdf_mod._candidate_urls(_row(conn, 1))) == ["http://s2/x.pdf"]


# --- fetch con fallback y retry ---

@pytest.fixture
def pdf_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PDF_DIR", tmp_path)
    return tmp_path


def _fake_download(respuestas: dict):
    """_download simulado: url -> bytes o None."""
    return lambda client, url: respuestas.get(url)


def test_fetch_prueba_candidatos_hasta_pdf_valido(conn, pdf_dir, monkeypatch):
    _add_paper(conn, id=1, pdf_url="http://roto/x.pdf", doi="10.1/x")
    monkeypatch.setattr(pdf_mod.unpaywall, "find_pdf_url", lambda doi: "http://up/x.pdf")
    monkeypatch.setattr(pdf_mod.semanticscholar, "find_pdf_url", lambda doi: None)
    monkeypatch.setattr(pdf_mod, "_download",
                        _fake_download({"http://roto/x.pdf": None, "http://up/x.pdf": b"%PDF-ok"}))
    r = pdf_mod.fetch_pdfs(conn)
    assert r["descargados"] == 1 and r["fallidos"] == 0
    row = _row(conn, 1)
    assert row["status"] == "fetched"
    assert (pdf_dir / "1.pdf").read_bytes() == b"%PDF-ok"


def test_fetch_sin_retry_ignora_papers_ya_procesados(conn, pdf_dir, monkeypatch):
    _add_paper(conn, id=1, status="summarized", pdf_url="http://oa/x.pdf")
    monkeypatch.setattr(pdf_mod, "_download", _fake_download({"http://oa/x.pdf": b"%PDF-ok"}))
    r = pdf_mod.fetch_pdfs(conn)
    assert r["pendientes"] == 0 and _row(conn, 1)["pdf_path"] is None


def test_retry_descarga_y_resetea_chunks_sin_tocar_status(conn, pdf_dir, monkeypatch):
    _add_paper(conn, id=1, status="summarized", pdf_url="http://oa/x.pdf")
    conn.execute("INSERT INTO chunks (paper_id, seq, text) VALUES (1, 0, 'solo abstract')")
    monkeypatch.setattr(pdf_mod, "_download", _fake_download({"http://oa/x.pdf": b"%PDF-ok"}))
    r = pdf_mod.fetch_pdfs(conn, retry=True)
    assert r["descargados"] == 1
    row = _row(conn, 1)
    assert row["status"] == "summarized"          # no regresa a 'fetched'
    assert row["pdf_path"] is not None
    assert conn.execute("SELECT COUNT(*) FROM chunks WHERE paper_id = 1").fetchone()[0] == 0
    # y el FTS quedó limpio para el re-indexado
    assert conn.execute("SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'abstract'").fetchone()[0] == 0


def test_retry_no_repite_papers_con_pdf(conn, pdf_dir, monkeypatch):
    _add_paper(conn, id=1, status="summarized", pdf_url="http://oa/x.pdf")
    conn.execute("UPDATE papers SET pdf_path = '/x/1.pdf' WHERE id = 1")
    monkeypatch.setattr(pdf_mod, "_download", _fake_download({}))
    assert pdf_mod.fetch_pdfs(conn, retry=True)["pendientes"] == 0


def test_fetch_cuenta_sin_url_y_fallidos(conn, pdf_dir, monkeypatch):
    _add_paper(conn, id=1)                                  # sin ninguna fuente
    _add_paper(conn, id=2, pdf_url="http://roto/x.pdf")     # todo falla
    monkeypatch.setattr(pdf_mod, "_download", _fake_download({}))
    r = pdf_mod.fetch_pdfs(conn)
    assert r == {"descargados": 0, "fallidos": 1, "sin_url": 1, "pendientes": 2}
