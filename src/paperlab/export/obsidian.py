"""Exportador de la biblioteca a un vault de Obsidian.

Genera una nota markdown por paper (frontmatter + resumen estructurado),
wikilinks entre papers a partir de la tabla `citations`, una nota MOC por
búsqueda guardada y un índice global. El export es idempotente: todo lo
anterior al MARCADOR se regenera en cada corrida y lo que el usuario escriba
debajo (`## Mis notas`) se preserva.
"""

import ast
import json
import re
import sqlite3
import sys
from pathlib import Path

from .. import db

MARKER = "%% paperlab: todo lo anterior se regenera en cada export. Escribe solo debajo. %%"
DEFAULT_USER_SECTION = "\n## Mis notas\n\n"

_ILLEGAL = re.compile(r'[\\/:*?"<>|\[\]#^]')
_FRONTMATTER_ID = re.compile(r"^paperlab_id:\s*(\d+)\s*$", re.MULTILINE)


def sanitize_filename(title: str) -> str:
    """Nombre seguro para cualquier SO y válido dentro de un wikilink."""
    clean = _ILLEGAL.sub(" ", title)
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean.strip(". ")


def build_note_names(papers: list[sqlite3.Row]) -> dict[int, str]:
    """paper_id -> nombre de nota (sin .md), determinista y sin colisiones.

    Los papers llegan ordenados por id (inmutable), así que ante títulos
    repetidos el sufijo con id siempre cae en los mismos papers.
    """
    names: dict[int, str] = {}
    used: set[str] = set()
    for p in papers:
        year = p["year"] if p["year"] else "s-f"
        base = f"{year} - {sanitize_filename(p['title'])[:80].strip('. ')}"
        name = base if base not in used else f"{base} (paperlab-{p['id']})"
        used.add(name)
        names[p["id"]] = name
    return names


def load_local_citations(conn: sqlite3.Connection) -> tuple[dict[int, list[int]], dict[int, int]]:
    """(paper_id -> ids citados presentes en la biblioteca, paper_id -> nº de refs externas)."""
    local_cites: dict[int, list[int]] = {}
    for row in conn.execute(
        """SELECT c.paper_id, p.id AS cited_id
           FROM citations c JOIN papers p ON p.openalex_id = c.cited_openalex_id
           WHERE p.id != c.paper_id
           ORDER BY c.paper_id, p.id"""
    ):
        local_cites.setdefault(row["paper_id"], []).append(row["cited_id"])
    external_refs: dict[int, int] = {}
    for row in conn.execute(
        """SELECT c.paper_id, COUNT(*) AS n
           FROM citations c LEFT JOIN papers p ON p.openalex_id = c.cited_openalex_id
           WHERE p.id IS NULL
           GROUP BY c.paper_id"""
    ):
        external_refs[row["paper_id"]] = row["n"]
    return local_cites, external_refs


def _clean_text(value: str | None) -> str:
    """Normaliza campos de resumen.

    Algunos resúmenes viejos guardan `str(lista)` de Python (el LLM devolvió una
    lista y analyze.py la stringificó), quedando como "['frase', 'frase']". Se
    detecta y se une en párrafo; el resto se devuelve tal cual.
    """
    if not value:
        return ""
    s = value.strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            items = ast.literal_eval(s)
            if isinstance(items, list):
                return " ".join(str(x).strip() for x in items if str(x).strip())
        except (ValueError, SyntaxError):
            pass
    return s


def _yaml_str(value: str) -> str:
    # Un string JSON es un string YAML válido: escapado gratis sin dependencia nueva.
    return json.dumps(value, ensure_ascii=False)


def render_frontmatter(paper: sqlite3.Row, summary: sqlite3.Row | None) -> str:
    aliases = []
    if paper["doi"]:
        aliases.append(f"doi:{paper['doi']}")
    if paper["arxiv_id"]:
        aliases.append(f"arXiv:{paper['arxiv_id']}")
    if paper["openalex_id"]:
        aliases.append(paper["openalex_id"])
    authors_list = json.loads(paper["authors"] or "[]")

    lines = ["---", f"paperlab_id: {paper['id']}", f"title: {_yaml_str(paper['title'])}"]
    if aliases:
        lines.append(f"aliases: [{', '.join(_yaml_str(a) for a in aliases)}]")
    if authors_list:
        lines.append(f"authors: [{', '.join(_yaml_str(a) for a in authors_list)}]")
    if paper["year"]:
        lines.append(f"year: {paper['year']}")
    for field in ("venue", "doi", "arxiv_id", "openalex_id", "url", "source"):
        if paper[field]:
            lines.append(f"{field}: {_yaml_str(paper[field])}")
    lines.append(f"estado: {_yaml_str(paper['status'])}")
    if paper["added_at"]:
        lines.append(f"añadido: {paper['added_at'][:10]}")
    if summary and summary["model"]:
        lines.append(f"modelo: {_yaml_str(summary['model'])}")
    lines.append("tags: [paper]")
    lines.append("---")
    return "\n".join(lines)


def render_note(
    paper: sqlite3.Row,
    summary: sqlite3.Row | None,
    cited_names: list[str],
    n_external: int,
    user_section: str,
) -> str:
    parts = [render_frontmatter(paper, summary), "", f"# {paper['title']}"]

    if summary:
        parts += ["", "## Resumen", "", _clean_text(summary["summary_md"])]
        findings_list = json.loads(summary["findings"] or "[]")
        if findings_list:
            parts += ["", "## Hallazgos", ""] + [f"- {h}" for h in findings_list]
        for title, field in (("Método", "method"), ("Limitaciones", "limitations"), ("Relevancia", "relevance")):
            text = _clean_text(summary[field])
            if text:
                parts += ["", f"## {title}", "", text]
    elif paper["abstract"]:
        parts += ["", "## Abstract", "", paper["abstract"].strip()]

    if cited_names or n_external:
        parts += ["", "## Citas", ""]
        parts += [f"- [[{n}]]" for n in cited_names]
        if n_external:
            if cited_names:
                parts.append("")
            parts.append(f"*(y {n_external} referencias externas fuera de la biblioteca)*")

    parts += ["", MARKER]
    return "\n".join(parts) + (user_section or DEFAULT_USER_SECTION)


def extract_user_section(path: Path) -> str | None:
    """Texto posterior al MARCADOR de una nota existente; None si no hay marcador."""
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return None
    if MARKER not in content:
        return None
    return content.split(MARKER, 1)[1]


def write_if_changed(path: Path, content: str) -> bool:
    """Escribe solo si el contenido cambió (evita churn de sync en LiveSync)."""
    try:
        if path.read_text(encoding="utf-8") == content:
            return False
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def _read_paperlab_id(path: Path) -> int | None:
    try:
        head = path.read_text(encoding="utf-8")[:2000]
    except OSError:
        return None
    if not head.startswith("---"):
        return None
    m = _FRONTMATTER_ID.search(head)
    return int(m.group(1)) if m else None


def papers_for_query(conn: sqlite3.Connection, query: str) -> list[int]:
    """Ids de papers locales que matchean la búsqueda (FTS5, por relevancia)."""
    try:
        rows = conn.execute(
            "SELECT rowid FROM papers_fts WHERE papers_fts MATCH ? ORDER BY rank",
            (db.fts_escape(query),),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [r["rowid"] for r in rows]


def render_moc(search_row: sqlite3.Row, names: list[str], user_section: str) -> str:
    lines = [
        "---",
        f"busqueda: {_yaml_str(search_row['query'])}",
        "tags: [moc]",
        "---",
        "",
        f"# MOC - {search_row['name']}",
        "",
        f"Papers locales que matchean `{search_row['query']}` ({len(names)}):",
        "",
    ]
    lines += [f"- [[{n}]]" for n in names]
    lines += ["", MARKER]
    return "\n".join(lines) + (user_section or DEFAULT_USER_SECTION)


def render_index(papers: list[sqlite3.Row], names: dict[int, str], user_section: str) -> str:
    lines = ["---", "tags: [moc]", "---", "", "# Índice de papers", ""]
    by_year: dict[object, list[sqlite3.Row]] = {}
    for p in papers:
        by_year.setdefault(p["year"], []).append(p)
    with_year = sorted((y for y in by_year if y), reverse=True)
    for y in with_year + ([None] if None in by_year else []):
        lines += [f"## {y or 'Sin fecha'}", ""]
        lines += [f"- [[{names[p['id']]}]]" for p in by_year[y]]
        lines.append("")
    lines += [MARKER]
    return "\n".join(lines) + (user_section or DEFAULT_USER_SECTION)


def export_vault(
    conn: sqlite3.Connection,
    vault: Path,
    prune: bool = False,
    dry_run: bool = False,
) -> dict:
    papers_dir = vault / "Papers"
    moc_dir = papers_dir / "MOC"

    papers = conn.execute("SELECT * FROM papers WHERE excluded = 0 ORDER BY id").fetchall()
    summaries = {r["paper_id"]: r for r in conn.execute("SELECT * FROM summaries")}
    names = build_note_names(papers)
    cites, external_refs = load_local_citations(conn)

    # Notas existentes gestionadas por paperlab (con paperlab_id en frontmatter).
    existing: dict[int, Path] = {}
    for f in sorted(papers_dir.glob("*.md")) if papers_dir.is_dir() else []:
        pid = _read_paperlab_id(f)
        if pid is not None:
            existing[pid] = f

    stats = {
        "notes": len(papers), "created": 0, "updated": 0, "unchanged": 0,
        "renamed": 0, "local_citations": sum(len(v) for v in cites.values()),
        "external_refs": sum(external_refs.values()), "mocs": 0,
        "orphans": [], "pruned": 0, "without_marker": [],
    }

    def _write(path: Path, content: str, existed: bool) -> None:
        try:
            current = path.read_text(encoding="utf-8")
        except OSError:
            current = None
        if current == content:
            stats["unchanged"] += 1
            return
        if not dry_run:
            write_if_changed(path, content)
        if existed:
            stats["updated"] += 1
        else:
            stats["created"] += 1

    for p in papers:
        target = papers_dir / f"{names[p['id']]}.md"
        previous = existing.get(p["id"])
        section = None
        if previous is not None:
            section = extract_user_section(previous)
            if section is None and previous.exists():
                # Sin marcador no se distingue lo gestionado de lo del usuario:
                # se preserva el cuerpo completo previo para no perder nada.
                stats["without_marker"].append(previous.name)
                old = previous.read_text(encoding="utf-8")
                body = old.split("---", 2)[2] if old.startswith("---") else old
                section = (
                    "\n## Mis notas\n\n"
                    "> [!warning] Nota previa sin marcador de paperlab; contenido preservado íntegro:\n\n"
                    + body.strip() + "\n"
                )
        content = render_note(
            p, summaries.get(p["id"]),
            [names[c] for c in cites.get(p["id"], [])],
            external_refs.get(p["id"], 0),
            section or "",
        )
        _write(target, content, existed=previous is not None)
        if previous is not None and previous != target:
            stats["renamed"] += 1
            if not dry_run:
                previous.unlink(missing_ok=True)

    # Huérfanas: notas gestionadas cuyo paper ya no está en la BD.
    db_ids = {p["id"] for p in papers}
    for pid, f in existing.items():
        if pid not in db_ids:
            if prune and not dry_run:
                f.unlink(missing_ok=True)
                stats["pruned"] += 1
            else:
                stats["orphans"].append(f.name)

    # MOCs por búsqueda guardada + índice global.
    for b in conn.execute("SELECT * FROM saved_searches ORDER BY id"):
        ids = [i for i in papers_for_query(conn, b["query"]) if i in names]
        target = moc_dir / f"MOC - {sanitize_filename(b['name'])}.md"
        content = render_moc(b, [names[i] for i in ids], extract_user_section(target) or "")
        _write(target, content, existed=target.exists())
        stats["mocs"] += 1

    target = moc_dir / "Índice de papers.md"
    content = render_index(papers, names, extract_user_section(target) or "")
    _write(target, content, existed=target.exists())
    stats["mocs"] += 1

    for name in stats["without_marker"]:
        print(f"⚠ {name}: sin marcador de paperlab; se regeneró sin descartar (revisar duplicados)", file=sys.stderr)
    return stats
