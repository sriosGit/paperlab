"""Tests del cliente WebDAV del NAS (httpx.MockTransport, sin red)."""

import sqlite3

import httpx
import pytest

from paperlab import config, db, nas


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript(db.SCHEMA)
    yield c
    c.close()


@pytest.fixture
def nas_config(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "NAS_BASE_URL", "http://nas:8000")
    monkeypatch.setattr(config, "NAS_USERNAME", "zerg")
    monkeypatch.setattr(config, "NAS_PASSWORD", "secreto")
    monkeypatch.setattr(config, "NAS_PDF_DIR", "paperlab/pdfs")
    monkeypatch.setattr(config, "PDF_DIR", tmp_path)
    return tmp_path


def _mock_client(monkeypatch, handler):
    def client():
        return httpx.Client(
            transport=httpx.MockTransport(handler),
            base_url="http://nas:8000/dav",
            auth=(config.NAS_USERNAME, config.NAS_PASSWORD),
        )
    monkeypatch.setattr(nas, "_client", client)


def _propfind_xml(nombres):
    hrefs = "".join(
        f"<D:response><D:href>/dav/paperlab/pdfs/{n}</D:href></D:response>" for n in nombres
    )
    return (
        '<?xml version="1.0"?><D:multistatus xmlns:D="DAV:">'
        f"<D:response><D:href>/dav/paperlab/pdfs/</D:href></D:response>{hrefs}"
        "</D:multistatus>"
    )


def _add_paper(conn, id, pdf_path=None):
    conn.execute(
        "INSERT INTO papers (id, title, title_norm, source, pdf_path, authors) VALUES (?, 'T', 't', 'arxiv', ?, '[]')",
        (id, pdf_path),
    )


def test_enabled_requiere_credenciales(monkeypatch):
    monkeypatch.setattr(config, "NAS_BASE_URL", "")
    assert not nas.enabled()
    monkeypatch.setattr(config, "NAS_BASE_URL", "http://nas:8000")
    monkeypatch.setattr(config, "NAS_USERNAME", "u")
    monkeypatch.setattr(config, "NAS_PASSWORD", "p")
    assert nas.enabled()


def test_sync_sube_solo_los_que_faltan(conn, nas_config, monkeypatch):
    local1 = nas_config / "1.pdf"; local1.write_bytes(b"%PDF-1")
    local2 = nas_config / "2.pdf"; local2.write_bytes(b"%PDF-2")
    _add_paper(conn, 1, str(local1))
    _add_paper(conn, 2, str(local2))
    peticiones = []

    def handler(request):
        peticiones.append((request.method, request.url.path))
        if request.method == "MKCOL":
            return httpx.Response(405)  # ya existían las carpetas
        if request.method == "PROPFIND":
            return httpx.Response(207, content=_propfind_xml(["2.pdf"]))
        if request.method == "PUT":
            assert request.content == b"%PDF-1"
            return httpx.Response(201)
        raise AssertionError(f"petición inesperada: {request.method}")

    _mock_client(monkeypatch, handler)
    r = nas.sync_pdfs(conn)
    assert r == {"subidos": 1, "ya_en_nas": 1, "recuperados": 0, "perdidos": 0}
    assert ("PUT", "/dav/paperlab/pdfs/1.pdf") in peticiones
    assert ("MKCOL", "/dav/paperlab") in peticiones  # crea las colecciones por segmento


def test_restore_recupera_y_actualiza_ruta(conn, nas_config, monkeypatch):
    _add_paper(conn, 3, None)  # sin PDF local, pero existe 3.pdf en el NAS

    def handler(request):
        if request.method == "MKCOL":
            return httpx.Response(405)
        if request.method == "PROPFIND":
            return httpx.Response(207, content=_propfind_xml(["3.pdf"]))
        if request.method == "GET":
            return httpx.Response(200, content=b"%PDF-3")
        raise AssertionError(request.method)

    _mock_client(monkeypatch, handler)
    r = nas.sync_pdfs(conn, restore=True)
    assert r["recuperados"] == 1
    assert (nas_config / "3.pdf").read_bytes() == b"%PDF-3"
    row = conn.execute("SELECT pdf_path FROM papers WHERE id = 3").fetchone()
    assert row["pdf_path"] == str(nas_config / "3.pdf")


def test_sin_restore_no_baja_ni_cuenta_perdido(conn, nas_config, monkeypatch):
    _add_paper(conn, 3, None)
    _add_paper(conn, 4, str(nas_config / "4.pdf"))  # la BD apunta a archivo borrado

    def handler(request):
        if request.method == "MKCOL":
            return httpx.Response(405)
        if request.method == "PROPFIND":
            return httpx.Response(207, content=_propfind_xml(["3.pdf"]))
        raise AssertionError(request.method)

    _mock_client(monkeypatch, handler)
    r = nas.sync_pdfs(conn)
    assert r == {"subidos": 0, "ya_en_nas": 0, "recuperados": 0, "perdidos": 1}


def test_carpeta_remota_inexistente_equivale_a_vacia(conn, nas_config, monkeypatch):
    local = nas_config / "1.pdf"; local.write_bytes(b"%PDF-1")
    _add_paper(conn, 1, str(local))

    def handler(request):
        if request.method == "MKCOL":
            return httpx.Response(201)
        if request.method == "PROPFIND":
            return httpx.Response(404)
        if request.method == "PUT":
            return httpx.Response(201)
        raise AssertionError(request.method)

    _mock_client(monkeypatch, handler)
    assert nas.sync_pdfs(conn)["subidos"] == 1


def test_credenciales_invalidas_dan_error_claro(conn, nas_config, monkeypatch):
    _mock_client(monkeypatch, lambda request: httpx.Response(401))
    with pytest.raises(nas.NasError, match="credenciales"):
        nas.sync_pdfs(conn)


def test_sync_sin_config_da_error(conn, monkeypatch):
    monkeypatch.setattr(config, "NAS_BASE_URL", "")
    with pytest.raises(nas.NasError, match="NAS_BASE_URL"):
        nas.sync_pdfs(conn)
