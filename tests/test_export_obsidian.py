"""Tests del exportador a Obsidian (I/O puro sobre SQLite in-memory + tmp_path)."""

import json
import sqlite3

import pytest

from paperlab import db
from paperlab.export import obsidian


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.row_factory = sqlite3.Row
    c.executescript(db.SCHEMA)
    yield c
    c.close()


def _add_paper(conn, *, id, title, year=2020, openalex_id=None, doi=None,
               arxiv_id=None, status="summarized", abstract="abs"):
    conn.execute(
        """INSERT INTO papers (id, title, title_norm, openalex_id, doi, arxiv_id,
                               year, source, status, abstract, authors, url)
           VALUES (?, ?, ?, ?, ?, ?, ?, 'openalex', ?, ?, ?, ?)""",
        (id, title, title.lower(), openalex_id, doi, arxiv_id, year, status,
         abstract, json.dumps(["A. Autora"]), "http://x"),
    )


def _add_summary(conn, paper_id, summary_md="Un resumen.", findings=None):
    conn.execute(
        """INSERT INTO summaries (paper_id, summary_md, findings, method, model)
           VALUES (?, ?, ?, 'un metodo', 'qwen2.5:14b')""",
        (paper_id, summary_md, json.dumps(findings or ["h1"])),
    )


# --- sanitización y nombres ---

def test_sanitize_quita_caracteres_ilegales():
    assert obsidian.sanitize_filename('A/B: [c] #x?') == "A B c x"
    assert obsidian.sanitize_filename("titulo.  ") == "titulo"


def test_build_note_names_resuelve_colisiones(conn):
    _add_paper(conn, id=1, title="Mismo título", year=2020)
    _add_paper(conn, id=2, title="Mismo título", year=2020)
    papers = conn.execute("SELECT * FROM papers ORDER BY id").fetchall()
    names = obsidian.build_note_names(papers)
    assert names[1] == "2020 - Mismo título"
    assert names[2] == "2020 - Mismo título (paperlab-2)"
    assert names[1] != names[2]


# --- citas locales vs externas ---

def test_wikilinks_solo_apuntan_a_papers_locales(conn):
    _add_paper(conn, id=1, title="Cita a otro", openalex_id="W1")
    _add_paper(conn, id=2, title="Citado local", openalex_id="W2")
    conn.execute("INSERT INTO citations VALUES (1, 'W2')")      # local
    conn.execute("INSERT INTO citations VALUES (1, 'W999')")    # externo
    local_cites, external_refs = obsidian.load_local_citations(conn)
    assert local_cites == {1: [2]}
    assert external_refs == {1: 1}


def test_clean_text_normaliza_lista_python():
    raw = "['frase uno', 'frase dos']"
    assert obsidian._clean_text(raw) == "frase uno frase dos"
    assert obsidian._clean_text("texto normal") == "texto normal"
    assert obsidian._clean_text(None) == ""


# --- export end to end sobre tmp_path ---

def test_export_crea_notas_mocs_e_indice(conn, tmp_path):
    _add_paper(conn, id=1, title="Paper uno", openalex_id="W1")
    _add_summary(conn, 1)
    conn.execute("INSERT INTO saved_searches (id, name, query) VALUES (1, 'ia', 'paper')")
    stats = obsidian.export_vault(conn, tmp_path)
    assert (tmp_path / "Papers" / "2020 - Paper uno.md").exists()
    assert (tmp_path / "Papers" / "MOC" / "MOC - ia.md").exists()
    assert (tmp_path / "Papers" / "MOC" / "Índice de papers.md").exists()
    assert stats["created"] == 3  # 1 paper + 1 moc + indice


def test_reexport_es_idempotente(conn, tmp_path):
    _add_paper(conn, id=1, title="Paper uno")
    _add_summary(conn, 1)
    obsidian.export_vault(conn, tmp_path)
    stats = obsidian.export_vault(conn, tmp_path)
    assert stats["created"] == 0
    assert stats["updated"] == 0
    assert stats["unchanged"] == 2  # paper + indice (no hay saved_searches)


def test_seccion_usuario_se_preserva(conn, tmp_path):
    _add_paper(conn, id=1, title="Paper uno")
    _add_summary(conn, 1)
    obsidian.export_vault(conn, tmp_path)
    nota = tmp_path / "Papers" / "2020 - Paper uno.md"
    content = nota.read_text(encoding="utf-8")
    nota.write_text(content + "Mi nota personal.\n", encoding="utf-8")

    obsidian.export_vault(conn, tmp_path)
    assert "Mi nota personal." in nota.read_text(encoding="utf-8")


def test_prune_borra_huerfanas(conn, tmp_path):
    _add_paper(conn, id=1, title="Paper uno")
    obsidian.export_vault(conn, tmp_path)
    conn.execute("DELETE FROM papers WHERE id=1")

    sin_prune = obsidian.export_vault(conn, tmp_path)
    assert len(sin_prune["orphans"]) == 1
    assert (tmp_path / "Papers" / "2020 - Paper uno.md").exists()

    con_prune = obsidian.export_vault(conn, tmp_path, prune=True)
    assert con_prune["pruned"] == 1
    assert not (tmp_path / "Papers" / "2020 - Paper uno.md").exists()


def test_rename_traslada_seccion_usuario(conn, tmp_path):
    _add_paper(conn, id=1, title="Título viejo")
    obsidian.export_vault(conn, tmp_path)
    vieja = tmp_path / "Papers" / "2020 - Título viejo.md"
    vieja.write_text(vieja.read_text(encoding="utf-8") + "Anotación.\n", encoding="utf-8")

    conn.execute("UPDATE papers SET title='Título nuevo' WHERE id=1")
    obsidian.export_vault(conn, tmp_path)

    nueva = tmp_path / "Papers" / "2020 - Título nuevo.md"
    assert nueva.exists()
    assert not vieja.exists()
    assert "Anotación." in nueva.read_text(encoding="utf-8")
