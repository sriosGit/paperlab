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

MARCADOR = "%% paperlab: todo lo anterior se regenera en cada export. Escribe solo debajo. %%"
SECCION_USUARIO_DEFAULT = "\n## Mis notas\n\n"

_ILEGALES = re.compile(r'[\\/:*?"<>|\[\]#^]')
_FRONTMATTER_ID = re.compile(r"^paperlab_id:\s*(\d+)\s*$", re.MULTILINE)


def sanitize_filename(titulo: str) -> str:
    """Nombre seguro para cualquier SO y válido dentro de un wikilink."""
    limpio = _ILEGALES.sub(" ", titulo)
    limpio = re.sub(r"\s+", " ", limpio).strip()
    return limpio.strip(". ")


def build_note_names(papers: list[sqlite3.Row]) -> dict[int, str]:
    """paper_id -> nombre de nota (sin .md), determinista y sin colisiones.

    Los papers llegan ordenados por id (inmutable), así que ante títulos
    repetidos el sufijo con id siempre cae en los mismos papers.
    """
    nombres: dict[int, str] = {}
    usados: set[str] = set()
    for p in papers:
        year = p["year"] if p["year"] else "s-f"
        base = f"{year} - {sanitize_filename(p['title'])[:80].strip('. ')}"
        nombre = base if base not in usados else f"{base} (paperlab-{p['id']})"
        usados.add(nombre)
        nombres[p["id"]] = nombre
    return nombres


def load_citas_locales(conn: sqlite3.Connection) -> tuple[dict[int, list[int]], dict[int, int]]:
    """(paper_id -> ids citados presentes en la biblioteca, paper_id -> nº de refs externas)."""
    locales: dict[int, list[int]] = {}
    for row in conn.execute(
        """SELECT c.paper_id, p.id AS cited_id
           FROM citations c JOIN papers p ON p.openalex_id = c.cited_openalex_id
           WHERE p.id != c.paper_id
           ORDER BY c.paper_id, p.id"""
    ):
        locales.setdefault(row["paper_id"], []).append(row["cited_id"])
    externas: dict[int, int] = {}
    for row in conn.execute(
        """SELECT c.paper_id, COUNT(*) AS n
           FROM citations c LEFT JOIN papers p ON p.openalex_id = c.cited_openalex_id
           WHERE p.id IS NULL
           GROUP BY c.paper_id"""
    ):
        externas[row["paper_id"]] = row["n"]
    return locales, externas


def _texto_limpio(valor: str | None) -> str:
    """Normaliza campos de resumen.

    Algunos resúmenes viejos guardan `str(lista)` de Python (el LLM devolvió una
    lista y analyze.py la stringificó), quedando como "['frase', 'frase']". Se
    detecta y se une en párrafo; el resto se devuelve tal cual.
    """
    if not valor:
        return ""
    s = valor.strip()
    if s.startswith("[") and s.endswith("]"):
        try:
            items = ast.literal_eval(s)
            if isinstance(items, list):
                return " ".join(str(x).strip() for x in items if str(x).strip())
        except (ValueError, SyntaxError):
            pass
    return s


def _yaml_str(valor: str) -> str:
    # Un string JSON es un string YAML válido: escapado gratis sin dependencia nueva.
    return json.dumps(valor, ensure_ascii=False)


def render_frontmatter(paper: sqlite3.Row, summary: sqlite3.Row | None) -> str:
    aliases = []
    if paper["doi"]:
        aliases.append(f"doi:{paper['doi']}")
    if paper["arxiv_id"]:
        aliases.append(f"arXiv:{paper['arxiv_id']}")
    if paper["openalex_id"]:
        aliases.append(paper["openalex_id"])
    autores = json.loads(paper["authors"] or "[]")

    lineas = ["---", f"paperlab_id: {paper['id']}", f"title: {_yaml_str(paper['title'])}"]
    if aliases:
        lineas.append(f"aliases: [{', '.join(_yaml_str(a) for a in aliases)}]")
    if autores:
        lineas.append(f"authors: [{', '.join(_yaml_str(a) for a in autores)}]")
    if paper["year"]:
        lineas.append(f"year: {paper['year']}")
    for campo in ("venue", "doi", "arxiv_id", "openalex_id", "url", "source"):
        if paper[campo]:
            lineas.append(f"{campo}: {_yaml_str(paper[campo])}")
    lineas.append(f"estado: {_yaml_str(paper['status'])}")
    if paper["added_at"]:
        lineas.append(f"añadido: {paper['added_at'][:10]}")
    if summary and summary["model"]:
        lineas.append(f"modelo: {_yaml_str(summary['model'])}")
    lineas.append("tags: [paper]")
    lineas.append("---")
    return "\n".join(lineas)


def render_note(
    paper: sqlite3.Row,
    summary: sqlite3.Row | None,
    nombres_citados: list[str],
    n_externas: int,
    seccion_usuario: str,
) -> str:
    partes = [render_frontmatter(paper, summary), "", f"# {paper['title']}"]

    if summary:
        partes += ["", "## Resumen", "", _texto_limpio(summary["summary_md"])]
        hallazgos = json.loads(summary["findings"] or "[]")
        if hallazgos:
            partes += ["", "## Hallazgos", ""] + [f"- {h}" for h in hallazgos]
        for titulo, campo in (("Método", "method"), ("Limitaciones", "limitations"), ("Relevancia", "relevance")):
            texto = _texto_limpio(summary[campo])
            if texto:
                partes += ["", f"## {titulo}", "", texto]
    elif paper["abstract"]:
        partes += ["", "## Abstract", "", paper["abstract"].strip()]

    if nombres_citados or n_externas:
        partes += ["", "## Citas", ""]
        partes += [f"- [[{n}]]" for n in nombres_citados]
        if n_externas:
            if nombres_citados:
                partes.append("")
            partes.append(f"*(y {n_externas} referencias externas fuera de la biblioteca)*")

    partes += ["", MARCADOR]
    return "\n".join(partes) + (seccion_usuario or SECCION_USUARIO_DEFAULT)


def extract_seccion_usuario(path: Path) -> str | None:
    """Texto posterior al MARCADOR de una nota existente; None si no hay marcador."""
    try:
        contenido = path.read_text(encoding="utf-8")
    except OSError:
        return None
    if MARCADOR not in contenido:
        return None
    return contenido.split(MARCADOR, 1)[1]


def write_if_changed(path: Path, contenido: str) -> bool:
    """Escribe solo si el contenido cambió (evita churn de sync en LiveSync)."""
    try:
        if path.read_text(encoding="utf-8") == contenido:
            return False
    except OSError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contenido, encoding="utf-8")
    return True


def _leer_paperlab_id(path: Path) -> int | None:
    try:
        cabeza = path.read_text(encoding="utf-8")[:2000]
    except OSError:
        return None
    if not cabeza.startswith("---"):
        return None
    m = _FRONTMATTER_ID.search(cabeza)
    return int(m.group(1)) if m else None


def papers_de_busqueda(conn: sqlite3.Connection, query: str) -> list[int]:
    """Ids de papers locales que matchean la búsqueda (FTS5, por relevancia)."""
    try:
        rows = conn.execute(
            "SELECT rowid FROM papers_fts WHERE papers_fts MATCH ? ORDER BY rank",
            (db.fts_escape(query),),
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [r["rowid"] for r in rows]


def render_moc(busqueda: sqlite3.Row, nombres: list[str], seccion_usuario: str) -> str:
    lineas = [
        "---",
        f"busqueda: {_yaml_str(busqueda['query'])}",
        "tags: [moc]",
        "---",
        "",
        f"# MOC - {busqueda['name']}",
        "",
        f"Papers locales que matchean `{busqueda['query']}` ({len(nombres)}):",
        "",
    ]
    lineas += [f"- [[{n}]]" for n in nombres]
    lineas += ["", MARCADOR]
    return "\n".join(lineas) + (seccion_usuario or SECCION_USUARIO_DEFAULT)


def render_indice(papers: list[sqlite3.Row], nombres: dict[int, str], seccion_usuario: str) -> str:
    lineas = ["---", "tags: [moc]", "---", "", "# Índice de papers", ""]
    por_year: dict[object, list[sqlite3.Row]] = {}
    for p in papers:
        por_year.setdefault(p["year"], []).append(p)
    con_year = sorted((y for y in por_year if y), reverse=True)
    for y in con_year + ([None] if None in por_year else []):
        lineas += [f"## {y or 'Sin fecha'}", ""]
        lineas += [f"- [[{nombres[p['id']]}]]" for p in por_year[y]]
        lineas.append("")
    lineas += [MARCADOR]
    return "\n".join(lineas) + (seccion_usuario or SECCION_USUARIO_DEFAULT)


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
    nombres = build_note_names(papers)
    citas, externas = load_citas_locales(conn)

    # Notas existentes gestionadas por paperlab (con paperlab_id en frontmatter).
    existentes: dict[int, Path] = {}
    for f in sorted(papers_dir.glob("*.md")) if papers_dir.is_dir() else []:
        pid = _leer_paperlab_id(f)
        if pid is not None:
            existentes[pid] = f

    stats = {
        "notas": len(papers), "nuevas": 0, "actualizadas": 0, "sin_cambios": 0,
        "renombradas": 0, "citas_locales": sum(len(v) for v in citas.values()),
        "refs_externas": sum(externas.values()), "mocs": 0,
        "huerfanas": [], "podadas": 0, "sin_marcador": [],
    }

    def _escribir(path: Path, contenido: str, existia: bool) -> None:
        try:
            actual = path.read_text(encoding="utf-8")
        except OSError:
            actual = None
        if actual == contenido:
            stats["sin_cambios"] += 1
            return
        if not dry_run:
            write_if_changed(path, contenido)
        if existia:
            stats["actualizadas"] += 1
        else:
            stats["nuevas"] += 1

    for p in papers:
        destino = papers_dir / f"{nombres[p['id']]}.md"
        previa = existentes.get(p["id"])
        seccion = None
        if previa is not None:
            seccion = extract_seccion_usuario(previa)
            if seccion is None and previa.exists():
                # Sin marcador no se distingue lo gestionado de lo del usuario:
                # se preserva el cuerpo completo previo para no perder nada.
                stats["sin_marcador"].append(previa.name)
                viejo = previa.read_text(encoding="utf-8")
                cuerpo = viejo.split("---", 2)[2] if viejo.startswith("---") else viejo
                seccion = (
                    "\n## Mis notas\n\n"
                    "> [!warning] Nota previa sin marcador de paperlab; contenido preservado íntegro:\n\n"
                    + cuerpo.strip() + "\n"
                )
        contenido = render_note(
            p, summaries.get(p["id"]),
            [nombres[c] for c in citas.get(p["id"], [])],
            externas.get(p["id"], 0),
            seccion or "",
        )
        _escribir(destino, contenido, existia=previa is not None)
        if previa is not None and previa != destino:
            stats["renombradas"] += 1
            if not dry_run:
                previa.unlink(missing_ok=True)

    # Huérfanas: notas gestionadas cuyo paper ya no está en la BD.
    ids_bd = {p["id"] for p in papers}
    for pid, f in existentes.items():
        if pid not in ids_bd:
            if prune and not dry_run:
                f.unlink(missing_ok=True)
                stats["podadas"] += 1
            else:
                stats["huerfanas"].append(f.name)

    # MOCs por búsqueda guardada + índice global.
    for b in conn.execute("SELECT * FROM saved_searches ORDER BY id"):
        ids = [i for i in papers_de_busqueda(conn, b["query"]) if i in nombres]
        destino = moc_dir / f"MOC - {sanitize_filename(b['name'])}.md"
        contenido = render_moc(b, [nombres[i] for i in ids], extract_seccion_usuario(destino) or "")
        _escribir(destino, contenido, existia=destino.exists())
        stats["mocs"] += 1

    destino = moc_dir / "Índice de papers.md"
    contenido = render_indice(papers, nombres, extract_seccion_usuario(destino) or "")
    _escribir(destino, contenido, existia=destino.exists())
    stats["mocs"] += 1

    for nombre in stats["sin_marcador"]:
        print(f"⚠ {nombre}: sin marcador de paperlab; se regeneró sin descartar (revisar duplicados)", file=sys.stderr)
    return stats
