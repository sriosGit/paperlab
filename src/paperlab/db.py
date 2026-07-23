"""SQLite: esquema, conexión y utilidades. Un solo archivo de datos.

Los embeddings se guardan como BLOB float32 en `chunks.embedding` y la
búsqueda vectorial se hace por fuerza bruta con numpy (suficiente hasta
decenas de miles de chunks; sqlite-vec queda como optimización futura).
"""

import sqlite3
from array import array

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    id INTEGER PRIMARY KEY,
    doi TEXT UNIQUE,
    arxiv_id TEXT UNIQUE,
    openalex_id TEXT UNIQUE,
    title TEXT NOT NULL,
    title_norm TEXT NOT NULL,
    abstract TEXT,
    authors TEXT NOT NULL DEFAULT '[]',   -- JSON
    year INTEGER,
    venue TEXT,
    source TEXT NOT NULL,
    url TEXT,
    pdf_url TEXT,
    pdf_path TEXT,
    status TEXT NOT NULL DEFAULT 'new',   -- new | fetched | indexed | summarized
    added_at TEXT NOT NULL DEFAULT (datetime('now')),
    -- descartado en la revisión: sigue en la BD (para no re-ingerirlo) pero
    -- queda fuera de resúmenes, RAG, síntesis y exports
    excluded INTEGER NOT NULL DEFAULT 0,
    excluded_reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_papers_title_norm ON papers(title_norm);
CREATE INDEX IF NOT EXISTS idx_papers_status ON papers(status);
CREATE INDEX IF NOT EXISTS idx_papers_excluded ON papers(excluded);

CREATE VIRTUAL TABLE IF NOT EXISTS papers_fts USING fts5(
    title, abstract, content='papers', content_rowid='id'
);
CREATE TRIGGER IF NOT EXISTS papers_ai AFTER INSERT ON papers BEGIN
    INSERT INTO papers_fts(rowid, title, abstract) VALUES (new.id, new.title, new.abstract);
END;
CREATE TRIGGER IF NOT EXISTS papers_ad AFTER DELETE ON papers BEGIN
    INSERT INTO papers_fts(papers_fts, rowid, title, abstract) VALUES ('delete', old.id, old.title, old.abstract);
END;
CREATE TRIGGER IF NOT EXISTS papers_au AFTER UPDATE OF title, abstract ON papers BEGIN
    INSERT INTO papers_fts(papers_fts, rowid, title, abstract) VALUES ('delete', old.id, old.title, old.abstract);
    INSERT INTO papers_fts(rowid, title, abstract) VALUES (new.id, new.title, new.abstract);
END;

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY,
    paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB,                        -- float32[] ; NULL = pendiente
    -- modelo que generó el embedding: vectores de modelos distintos tienen
    -- dimensiones distintas y no se pueden comparar entre sí
    embed_model TEXT,
    UNIQUE(paper_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_chunks_paper ON chunks(paper_id);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, content='chunks', content_rowid='id'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;

CREATE TABLE IF NOT EXISTS summaries (
    paper_id INTEGER PRIMARY KEY REFERENCES papers(id) ON DELETE CASCADE,
    summary_md TEXT NOT NULL,
    findings TEXT NOT NULL DEFAULT '[]',   -- JSON: lista de hallazgos
    method TEXT,
    limitations TEXT,
    relevance TEXT,
    model TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS extractions (
    paper_id INTEGER PRIMARY KEY REFERENCES papers(id) ON DELETE CASCADE,
    fields TEXT NOT NULL,                  -- JSON
    model TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS citations (
    paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    cited_openalex_id TEXT NOT NULL,
    UNIQUE(paper_id, cited_openalex_id)
);

CREATE TABLE IF NOT EXISTS syntheses (
    id INTEGER PRIMARY KEY,
    topic TEXT,                            -- NULL = corpus reciente sin filtrar
    paper_ids TEXT NOT NULL,               -- JSON: ids en el orden de cita [n]
    sections TEXT NOT NULL,                -- JSON: panorama, tendencias, contradicciones…
    model TEXT NOT NULL,
    group_id INTEGER REFERENCES groups(id) ON DELETE SET NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS saved_searches (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    query TEXT NOT NULL,
    sources TEXT NOT NULL DEFAULT 'arxiv,openalex',
    max_results INTEGER NOT NULL DEFAULT 50,
    from_year INTEGER,
    to_year INTEGER,
    last_run_at TEXT
);

-- Procedencia: qué búsqueda trajo cada paper. Se guarda la query textual
-- además del id para que el rastro sobreviva al borrado de la búsqueda.
CREATE TABLE IF NOT EXISTS paper_sources (
    paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    search_id INTEGER REFERENCES saved_searches(id) ON DELETE SET NULL,
    query TEXT NOT NULL,
    source TEXT NOT NULL,
    added_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(paper_id, query, source)
);
CREATE INDEX IF NOT EXISTS idx_paper_sources_paper ON paper_sources(paper_id);

-- Grupos: espacio persistente que junta búsquedas, papers y sus síntesis.
CREATE TABLE IF NOT EXISTS groups (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS group_papers (
    group_id INTEGER NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
    paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    added_via TEXT NOT NULL DEFAULT 'manual',  -- manual | search | citation
    added_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (group_id, paper_id)
);
CREATE INDEX IF NOT EXISTS idx_group_papers_paper ON group_papers(paper_id);

CREATE TABLE IF NOT EXISTS group_searches (
    group_id INTEGER NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
    search_id INTEGER NOT NULL REFERENCES saved_searches(id) ON DELETE CASCADE,
    PRIMARY KEY (group_id, search_id)
);
"""
# Este esquema es la única fuente de verdad: no hay migraciones. Si una base
# antigua no lo cumple, bórrala y vuelve a ingerir (`rm data/paperlab.db`).


def get_conn() -> sqlite3.Connection:
    config.ensure_dirs()
    conn = sqlite3.connect(config.DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    return conn


def embedding_to_blob(vec: list[float]) -> bytes:
    return array("f", vec).tobytes()


def blob_to_embedding(blob: bytes) -> list[float]:
    a = array("f")
    a.frombytes(blob)
    return list(a)


def fts_escape(query: str) -> str:
    """Convierte texto libre en una consulta FTS5 segura (términos entre comillas)."""
    terms = [t.replace('"', "") for t in query.split()]
    return " ".join(f'"{t}"' for t in terms if t)
