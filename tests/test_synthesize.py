"""Tests del análisis transversal (LLM simulado con monkeypatch)."""

import json
import sqlite3

import pytest

from paperlab import db, synthesize


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript(db.SCHEMA)
    yield c
    c.close()


def _add_paper(conn, *, id, title, year=2020, abstract="abs", added_at="2024-01-01"):
    conn.execute(
        """INSERT INTO papers (id, title, title_norm, year, source, abstract, authors, added_at)
           VALUES (?, ?, ?, ?, 'arxiv', ?, '[]', ?)""",
        (id, title, title.lower(), year, abstract, added_at),
    )


def _add_summary(conn, paper_id, findings=None, method="método X", limitations="pocos datos"):
    conn.execute(
        """INSERT INTO summaries (paper_id, summary_md, findings, method, limitations, model)
           VALUES (?, 'Un resumen.', ?, ?, ?, 'qwen2.5:14b')""",
        (paper_id, json.dumps(findings or ["h1"], ensure_ascii=False), method, limitations),
    )


# --- selección de papers ---

def test_select_por_tema_solo_papers_con_resumen(conn):
    _add_paper(conn, id=1, title="Grafeno y baterías")
    _add_paper(conn, id=2, title="Grafeno flexible")
    _add_paper(conn, id=3, title="Otra cosa")
    _add_summary(conn, 1)
    _add_summary(conn, 3)  # el 2 no tiene resumen
    ids = synthesize.select_papers(conn, "grafeno", limit=10)
    assert ids == [1]


def test_select_por_tema_cae_a_fts_de_chunks(conn):
    _add_paper(conn, id=1, title="Título sin la palabra", abstract=None)
    _add_summary(conn, 1)
    conn.execute("INSERT INTO chunks (paper_id, seq, text) VALUES (1, 0, 'habla de perovskita aquí')")
    ids = synthesize.select_papers(conn, "perovskita", limit=10)
    assert ids == [1]


def test_select_sin_tema_devuelve_resumidos_recientes(conn):
    _add_paper(conn, id=1, title="Viejo", added_at="2024-01-01")
    _add_paper(conn, id=2, title="Nuevo", added_at="2024-06-01")
    _add_paper(conn, id=3, title="Sin resumen", added_at="2024-12-01")
    _add_summary(conn, 1)
    _add_summary(conn, 2)
    assert synthesize.select_papers(conn, None, limit=10) == [2, 1]
    assert synthesize.select_papers(conn, None, limit=1) == [2]


# --- dossiers ---

def test_build_dossiers_numera_y_lista_fuentes(conn):
    _add_paper(conn, id=1, title="Uno")
    _add_paper(conn, id=2, title="Dos", year=2023)
    _add_summary(conn, 1, findings=["hallazgo A", "hallazgo B"])
    _add_summary(conn, 2)
    text, sources = synthesize.build_dossiers(conn, [2, 1])
    assert "[1] Dos (2023)" in text
    assert "[2] Uno (2020)" in text
    assert "hallazgo A; hallazgo B" in text
    assert [s["n"] for s in sources] == [1, 2]
    assert [s["paper_id"] for s in sources] == [2, 1]


def test_build_dossiers_respeta_presupuesto(conn, monkeypatch):
    for i in range(1, 6):
        _add_paper(conn, id=i, title=f"Paper {i}")
        _add_summary(conn, i, method="m" * 200)
    monkeypatch.setattr(synthesize, "DOSSIER_BUDGET", 600)
    _text, sources = synthesize.build_dossiers(conn, [1, 2, 3, 4, 5])
    assert synthesize.MIN_PAPERS <= len(sources) < 5


# --- run ---

def test_run_guarda_y_normaliza(conn, monkeypatch):
    _add_paper(conn, id=1, title="Tema uno")
    _add_paper(conn, id=2, title="Tema dos")
    _add_summary(conn, 1)
    _add_summary(conn, 2)
    respuesta = {
        "panorama": ["Campo activo [1][2].", "Crece rápido [1]."],
        "tendencias": ["más datos [1]"],
        "contradicciones": {"raro": "no-lista"},   # el modelo a veces no da lista
        "huecos": ["falta X [2]", {"detalle": "item no-string"}],
        # sin "consensos" ni "aplicaciones": deben quedar como lista vacía
    }
    monkeypatch.setattr(synthesize.llm, "generate_json", lambda *a, **k: respuesta)
    s = synthesize.run(conn, topic="tema")
    assert s.id == 1
    assert s.sections["panorama"] == "Campo activo [1][2]. Crece rápido [1]."
    assert s.sections["tendencias"] == ["más datos [1]"]
    assert s.sections["contradicciones"] == ['{"raro": "no-lista"}']
    assert s.sections["huecos"][0] == "falta X [2]"
    assert s.sections["consensos"] == [] and s.sections["aplicaciones"] == []

    guardada = synthesize.get(conn, s.id)
    assert guardada.topic == "tema"
    assert [x["paper_id"] for x in guardada.sources] == [x["paper_id"] for x in s.sources]
    assert guardada.sections == s.sections


def test_run_exige_minimo_de_resumenes(conn):
    _add_paper(conn, id=1, title="Solo uno")
    _add_summary(conn, 1)
    with pytest.raises(ValueError, match="al menos 2"):
        synthesize.run(conn)


# --- map-reduce ---

def test_run_full_numera_global_y_reduce(conn, monkeypatch):
    for i in range(1, 7):
        _add_paper(conn, id=i, title=f"Paper {i}", added_at=f"2024-0{i}-01")
        _add_summary(conn, i)
    prompts = []

    def fake(prompt, system=None):
        prompts.append(prompt)
        if "ANÁLISIS PARCIALES" in prompt:
            return {"panorama": "Global [1][6].", "tendencias": ["t global [3]"]}
        return {"panorama": "Parcial.", "tendencias": ["t parcial"]}

    monkeypatch.setattr(synthesize.llm, "generate_json", fake)
    s = synthesize.run_full(conn, batch_size=3)

    maps = [p for p in prompts if "ANÁLISIS PARCIALES" not in p]
    reduces = [p for p in prompts if "ANÁLISIS PARCIALES" in p]
    assert len(maps) == 2 and len(reduces) == 1
    assert "[1]" in maps[0] and "[3]" in maps[0] and "[4]" not in maps[0]
    assert "[4]" in maps[1] and "[6]" in maps[1]          # numeración global
    assert "papers [1]–[3]" in reduces[0] and "papers [4]–[6]" in reduces[0]
    assert s.sections["panorama"] == "Global [1][6]."     # gana el resultado del reduce
    assert [x["n"] for x in s.sources] == [1, 2, 3, 4, 5, 6]
    assert synthesize.get(conn, s.id).sections == s.sections


def test_run_full_un_solo_lote_no_reduce(conn, monkeypatch):
    for i in range(1, 4):
        _add_paper(conn, id=i, title=f"Paper {i}")
        _add_summary(conn, i)
    prompts = []

    def fake(prompt, system=None):
        prompts.append(prompt)
        return {"panorama": "Único lote."}

    monkeypatch.setattr(synthesize.llm, "generate_json", fake)
    s = synthesize.run_full(conn, batch_size=10)
    assert len(prompts) == 1 and "ANÁLISIS PARCIALES" not in prompts[0]
    assert s.sections["panorama"] == "Único lote."


def test_reduce_recursivo_agrupa_por_rondas(monkeypatch):
    llamadas = []

    def fake(prompt, system=None):
        llamadas.append(prompt)
        return {"panorama": "combinado"}

    monkeypatch.setattr(synthesize.llm, "generate_json", fake)
    partials = [
        {"first": i * 10 + 1, "last": i * 10 + 10, "sections": {"panorama": f"p{i}"}}
        for i in range(8)
    ]
    # 8 parciales con grupos de 6: ronda 1 = [6, 2] → 2 llamadas; ronda 2 = 1 llamada
    sections = synthesize._reduce(partials, None, lambda m: None)
    assert len(llamadas) == 3
    assert sections["panorama"] == "combinado"


def test_get_inexistente_y_fuente_borrada(conn, monkeypatch):
    assert synthesize.get(conn, 99) is None
    _add_paper(conn, id=1, title="Uno")
    _add_paper(conn, id=2, title="Dos")
    _add_summary(conn, 1)
    _add_summary(conn, 2)
    monkeypatch.setattr(synthesize.llm, "generate_json", lambda *a, **k: {})
    s = synthesize.run(conn)
    conn.execute("DELETE FROM papers WHERE id = 2")
    guardada = synthesize.get(conn, s.id)
    titulos = [x["title"] for x in guardada.sources]
    assert "Uno" in titulos
    assert any("borrado" in t for t in titulos)
