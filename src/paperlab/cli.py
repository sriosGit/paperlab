"""CLI de paperlab (typer)."""

import json
from pathlib import Path

import typer

from . import analyze, config, db, groups, llm, query_builder, review, synthesize
from . import pdf as pdf_mod
from .ingest import run_search

app = typer.Typer(help="Recopilador y analizador local de artículos científicos.", no_args_is_help=True)


@app.command()
def search(
    query: str,
    source: str = typer.Option("arxiv,openalex", help="Fuentes separadas por coma"),
    limit: int = typer.Option(50, help="Máximo de resultados por fuente"),
):
    """Busca en las fuentes y guarda los papers en la base de datos."""
    conn = db.get_conn()
    result = run_search(conn, query, source.split(","), limit)
    for src, info in result.items():
        typer.echo(f"{src}: {info}")


@app.command(name="search-ai")
def search_ai(
    request: str,
    limit: int = typer.Option(50, help="Máximo de resultados por fuente"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Ejecuta sin pedir confirmación"),
):
    """Traduce una petición en lenguaje natural a una query óptima (LLM local) y busca."""
    conn = db.get_conn()
    try:
        plan = query_builder.suggest(request)
    except llm.OllamaError as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(1)
    typer.echo(f"Query generada: {plan.query}")
    typer.echo(f"Fuentes: {', '.join(plan.sources)}")
    if plan.from_year or plan.to_year:
        typer.echo(f"Rango de años: {plan.from_year or '…'}–{plan.to_year or '…'}")
    if plan.notes:
        typer.echo(f"Nota: {plan.notes}")
    if not yes and not typer.confirm("¿Ejecutar esta búsqueda?", default=True):
        raise typer.Exit()
    result = run_search(
        conn, plan.query, plan.sources, limit, from_year=plan.from_year, to_year=plan.to_year
    )
    for src, info in result.items():
        typer.echo(f"{src}: {info}")


@app.command(name="fetch-pdfs")
def fetch_pdfs(
    limit: int = typer.Option(None, help="Máximo de PDFs a descargar"),
    retry: bool = typer.Option(False, "--retry", help="Reintenta todo paper sin PDF (no solo los 'new')"),
):
    """Descarga PDFs: URL guardada → arXiv → Unpaywall → Semantic Scholar."""
    conn = db.get_conn()
    typer.echo(json.dumps(pdf_mod.fetch_pdfs(conn, limit, retry=retry), ensure_ascii=False))


@app.command(name="relocate-pdfs")
def relocate_pdfs():
    """Mueve los PDFs descargados al directorio actual (PAPERLAB_PDF_DIR) y actualiza rutas."""
    import shutil

    conn = db.get_conn()
    config.ensure_dirs()
    rows = conn.execute("SELECT id, pdf_path FROM papers WHERE pdf_path IS NOT NULL").fetchall()
    moved = already_there = missing = 0
    for row in rows:
        current = Path(row["pdf_path"])
        target = config.PDF_DIR / current.name
        if current == target:
            already_there += 1
            continue
        if current.exists():
            shutil.move(str(current), target)
        elif not target.exists():
            typer.echo(f"  paper {row['id']}: PDF perdido ({current})", err=True)
            missing += 1
            continue
        conn.execute("UPDATE papers SET pdf_path = ? WHERE id = ?", (str(target), row["id"]))
        moved += 1
    conn.commit()
    typer.echo(f"destino: {config.PDF_DIR}")
    typer.echo(f"movidos: {moved} · ya en sitio: {already_there} · perdidos: {missing}")


@app.command()
def process(no_embed: bool = typer.Option(False, "--no-embed", help="Solo trocear e indexar FTS, sin embeddings")):
    """Extrae texto, trocea, indexa (FTS5) y calcula embeddings pendientes."""
    conn = db.get_conn()
    n = analyze.ensure_chunks(conn)
    typer.echo(f"papers troceados: {n}")
    if no_embed:
        return
    stale = analyze.stale_embeddings(conn)
    if stale:
        typer.echo(
            f"↻ {stale} chunks tienen embeddings de otro modelo: "
            f"se recalculan con {config.OLLAMA_EMBED_MODEL}"
        )
    try:
        e = analyze.ensure_embeddings(conn)
        typer.echo(f"chunks con embedding nuevo: {e}")
    except llm.OllamaError as exc:
        typer.echo(f"⚠ embeddings omitidos: {exc}", err=True)
        raise typer.Exit(1)


@app.command()
def summarize(
    limit: int = typer.Option(None, help="Máximo de papers a resumir"),
    paper_id: int = typer.Option(None, help="Resumir solo este paper"),
):
    """Genera resúmenes estructurados con el LLM local."""
    conn = db.get_conn()
    ids = [paper_id] if paper_id else analyze.pending_summaries(conn)
    if limit:
        ids = ids[:limit]
    if not ids:
        typer.echo("no hay papers pendientes de resumir")
        return
    for i, pid in enumerate(ids, 1):
        try:
            data = analyze.summarize_paper(conn, pid)
            typer.echo(f"[{i}/{len(ids)}] paper {pid}: {str(data.get('resumen', ''))[:100]}…")
        except (llm.OllamaError, ValueError) as exc:
            typer.echo(f"[{i}/{len(ids)}] paper {pid}: ERROR {exc}", err=True)


@app.command()
def ask(question: str, k: int = typer.Option(8, help="Chunks a recuperar")):
    """Pregunta al corpus (RAG con citas)."""
    conn = db.get_conn()
    try:
        answer = analyze.ask(conn, question, k=k)
    except llm.OllamaError as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(1)
    typer.echo(answer.text)
    if answer.sources:
        typer.echo("\nFuentes:")
        for s in answer.sources:
            typer.echo(f"  [{s['n']}] {s['title']} ({s['year'] or 's.f.'}) — paper #{s['paper_id']}")


def _print_synthesis(s: "synthesize.Synthesis") -> None:
    for key, label in synthesize.SECTIONS:
        value = s.sections.get(key)
        if not value:
            continue
        typer.echo(f"\n## {label}")
        if isinstance(value, str):
            typer.echo(value)
        else:
            for item in value:
                typer.echo(f"  - {item}")
    typer.echo("\nPapers comparados:")
    for src in s.sources:
        typer.echo(f"  [{src['n']}] {src['title']} ({src['year'] or 's.f.'}) — paper #{src['paper_id']}")
    a = s.audit
    if a:
        typer.echo(
            f"\nCitas: {len(a.cited)}/{a.n_sources} fuentes citadas "
            f"({a.coverage:.0%} de cobertura)"
        )
        if a.out_of_range:
            typer.echo(
                f"  ⚠ citas a fuentes inexistentes: {a.out_of_range}", err=True
            )
        if a.sections_without_citations:
            typer.echo(
                f"  ⚠ secciones sin ninguna cita: {', '.join(a.sections_without_citations)}", err=True
            )


@app.command(name="synthesize")
def synthesize_cmd(
    topic: str = typer.Argument(None, help="Tema de enfoque (vacío = resumidos más recientes)"),
    limit: int = typer.Option(synthesize.DEFAULT_LIMIT, help="Máximo de papers a comparar"),
    full: bool = typer.Option(False, "--full", help="Todo el corpus por lotes (map-reduce); --limit pasa a ser el tamaño de lote"),
    show: int = typer.Option(None, help="Mostrar una síntesis guardada por id (no genera)"),
    list_all: bool = typer.Option(False, "--list", help="Listar síntesis guardadas"),
):
    """Análisis transversal: compara papers y detecta tendencias, contradicciones y huecos."""
    conn = db.get_conn()
    if list_all:
        for r in synthesize.list_all(conn):
            n = len(json.loads(r["paper_ids"]))
            typer.echo(f"#{r['id']}  {r['created_at']}  {r['topic'] or '(sin tema)'}  — {n} papers")
        return
    if show is not None:
        s = synthesize.get(conn, show)
        if not s:
            typer.echo(f"ERROR: no existe la síntesis #{show}", err=True)
            raise typer.Exit(1)
        typer.echo(f"Síntesis #{s.id} — {s.topic or '(sin tema)'} — {s.created_at}")
        _print_synthesis(s)
        return
    try:
        if full:
            s = synthesize.run_full(conn, topic=topic, batch_size=limit, progress=typer.echo)
        else:
            s = synthesize.run(conn, topic=topic, limit=limit)
    except (llm.OllamaError, ValueError) as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(1)
    typer.echo(f"Síntesis #{s.id} — {s.topic or '(sin tema)'} — {len(s.sources)} papers")
    _print_synthesis(s)


@app.command()
def stats():
    """Muestra el estado del corpus."""
    conn = db.get_conn()
    total = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    by_status = conn.execute(
        "SELECT status, COUNT(*) FROM papers GROUP BY status"
    ).fetchall()
    chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    embedded = conn.execute("SELECT COUNT(*) FROM chunks WHERE embedding IS NOT NULL").fetchone()[0]
    summaries = conn.execute("SELECT COUNT(*) FROM summaries").fetchone()[0]
    excluidos = conn.execute("SELECT COUNT(*) FROM papers WHERE excluded = 1").fetchone()[0]
    typer.echo(
        f"papers: {total} (activos: {total - excluidos}, excluidos: {excluidos})  "
        + "  ".join(f"{r[0]}={r[1]}" for r in by_status)
    )
    typer.echo(f"chunks: {chunks} (con embedding: {embedded})")
    for r in conn.execute(
        "SELECT COALESCE(embed_model, '(desconocido)') m, COUNT(*) n FROM chunks "
        "WHERE embedding IS NOT NULL GROUP BY m"
    ):
        mark = "" if r["m"] == config.OLLAMA_EMBED_MODEL else "  ← obsoleto, se recalculará"
        typer.echo(f"  embeddings de {r['m']}: {r['n']}{mark}")
    typer.echo(f"resúmenes: {summaries}")
    typer.echo(f"ollama disponible: {'sí' if llm.is_available() else 'NO'}")


def _parse_ids(ids: str) -> list[int]:
    return [int(x) for x in ids.replace(",", " ").split() if x.strip()]


@app.command()
def exclude(
    ids: str = typer.Argument(..., help="Ids de papers separados por coma o espacio"),
    reason: str = typer.Option("", help="Motivo (se guarda para saber por qué se descartó)"),
):
    """Descarta papers del corpus: siguen en la BD pero fuera de análisis y exports."""
    conn = db.get_conn()
    n = review.exclude(conn, _parse_ids(ids), reason)
    typer.echo(f"excluidos: {n}")


@app.command()
def include(ids: str = typer.Argument(..., help="Ids de papers separados por coma o espacio")):
    """Reincorpora papers excluidos al corpus."""
    conn = db.get_conn()
    typer.echo(f"reincorporados: {review.include(conn, _parse_ids(ids))}")


@app.command(name="review")
def review_cmd(
    query: str = typer.Option(None, help="Ver los papers que trajo esta query"),
    only_from_this: bool = typer.Option(False, "--only", help="Con --query: solo los que ninguna otra búsqueda trajo"),
    excluded: bool = typer.Option(False, "--excluded", help="Listar los papers ya excluidos"),
):
    """Revisa la procedencia del corpus para detectar y descartar ruido."""
    conn = db.get_conn()
    if excluded:
        rows = conn.execute(
            "SELECT id, title, excluded_reason FROM papers WHERE excluded = 1 ORDER BY id"
        ).fetchall()
        for r in rows:
            reason = f" — {r['excluded_reason']}" if r["excluded_reason"] else ""
            typer.echo(f"  #{r['id']} {r['title'][:80]}{reason}")
        typer.echo(f"total excluidos: {len(rows)}")
        return
    if query:
        rows = review.by_query(conn, query, only_from_this=only_from_this)
        for r in rows:
            mark = "✗" if r["excluded"] else " "
            typer.echo(f" {mark} #{r['id']} {r['title'][:80]}")
        typer.echo(f"total: {len(rows)} papers de «{query}»")
        typer.echo("descarta con: paperlab exclude <ids> --reason «fuera de tema»")
        return
    s = review.stats(conn)
    typer.echo(
        f"papers: {s['total']} · activos: {s['active']} · excluidos: {s['excluded']} · "
        f"sin procedencia registrada: {s['without_provenance']}"
    )
    typer.echo("\nPor búsqueda:")
    for r in review.queries(conn):
        typer.echo(f"  {r['n_papers']:4d} papers ({r['n_excluded']} excluidos)  «{r['query']}»")


@app.command(name="enrich-openalex")
def enrich_openalex(limit: int = typer.Option(None, help="Máximo de papers a enriquecer")):
    """Rellena openalex_id y citas de papers que solo tienen DOI/arXiv (densifica el grafo)."""
    from .ingest import openalex
    from .ingest.base import store_papers

    conn = db.get_conn()
    pending = conn.execute(
        """SELECT id, doi, arxiv_id FROM papers
           WHERE openalex_id IS NULL AND (doi IS NOT NULL OR arxiv_id IS NOT NULL)
             AND excluded = 0
           ORDER BY id"""
    ).fetchall()
    if limit:
        pending = pending[:limit]
    if not pending:
        typer.echo("no hay papers con DOI/arXiv pendientes de enriquecer")
        return

    cites_before = conn.execute("SELECT COUNT(*) FROM citations").fetchone()[0]
    enriched = failed = 0
    for i, row in enumerate(pending, 1):
        try:
            paper = openalex.fetch_by_ids(row["doi"], row["arxiv_id"])
        except Exception as exc:  # noqa: BLE001 — un paper no debe tumbar el lote
            typer.echo(f"[{i}/{len(pending)}] paper {row['id']}: ERROR {exc}", err=True)
            failed += 1
            continue
        if paper and paper.openalex_id:
            store_papers(conn, [paper])
            enriched += 1
        else:
            failed += 1
        typer.echo(f"[{i}/{len(pending)}] paper {row['id']}: {'ok' if paper else 'sin match'}")

    cites_after = conn.execute("SELECT COUNT(*) FROM citations").fetchone()[0]
    typer.echo(
        f"enriquecidos: {enriched} · sin match/error: {failed} · "
        f"citas nuevas: {cites_after - cites_before}"
    )


@app.command(name="export-obsidian")
def export_obsidian(
    vault: Path = typer.Option(None, help="Carpeta del vault (por defecto $OBSIDIAN_VAULT_PATH)"),
    prune: bool = typer.Option(False, help="Borra notas de papers que ya no están en la BD"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Muestra qué haría sin escribir"),
):
    """Exporta la biblioteca al vault de Obsidian (notas, wikilinks de citas y MOCs)."""
    from .export import obsidian

    target = vault or config.OBSIDIAN_VAULT_PATH
    if target is None:
        typer.echo("ERROR: define OBSIDIAN_VAULT_PATH en .env o pasa --vault", err=True)
        raise typer.Exit(1)
    conn = db.get_conn()
    s = obsidian.export_vault(conn, target, prune=prune, dry_run=dry_run)
    prefix = "[dry-run] " if dry_run else ""
    typer.echo(
        f"{prefix}papers: {s['notes']} · MOCs: {s['mocs']} · "
        f"archivos nuevos {s['created']}, actualizados {s['updated']}, "
        f"sin cambios {s['unchanged']}, renombrados {s['renamed']} · "
        f"citas locales: {s['local_citations']} · refs externas: {s['external_refs']} · "
        f"huérfanas: {len(s['orphans'])} · podadas: {s['pruned']}"
    )
    for h in s["orphans"]:
        typer.echo(f"  huérfana (usa --prune para borrar): {h}")


@app.command(name="sync-nas")
def sync_nas(
    restore: bool = typer.Option(False, "--restore", help="Además, baja del NAS los PDFs que falten en disco"),
):
    """Respalda los PDFs en el NAS (SRC Cloud) vía WebDAV; con --restore también recupera."""
    from . import nas

    conn = db.get_conn()
    try:
        r = nas.sync_pdfs(conn, restore=restore, progress=typer.echo)
    except nas.NasError as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(1)
    typer.echo(
        f"subidos: {r['uploaded']} · ya en NAS: {r['already_on_nas']} · "
        f"recuperados: {r['restored']} · perdidos: {r['missing']}"
    )


group_app = typer.Typer(help="Grupos: junta búsquedas, papers y sus síntesis en un mismo espacio.")
app.add_typer(group_app, name="group")


@group_app.command(name="create")
def group_create(name: str, description: str = typer.Option("", help="Descripción opcional")):
    """Crea un grupo."""
    conn = db.get_conn()
    gid = groups.create(conn, name, description)
    typer.echo(f"grupo #{gid} creado: {name}")


@group_app.command(name="list")
def group_list():
    """Lista los grupos."""
    for g in groups.list_all(db.get_conn()):
        typer.echo(f"#{g['id']}  {g['name']}  — {g['n_papers']} papers, {g['n_searches']} búsquedas ligadas")


@group_app.command(name="link-search")
def group_link_search(group_id: int, search_id: int):
    """Liga una búsqueda guardada a un grupo (y sincroniza sus papers ya traídos)."""
    conn = db.get_conn()
    groups.link_search(conn, group_id, search_id)
    typer.echo(f"búsqueda #{search_id} ligada al grupo #{group_id}")


@group_app.command(name="add-paper")
def group_add_paper(group_id: int, paper_id: int):
    """Añade un paper suelto a un grupo."""
    conn = db.get_conn()
    groups.add_paper(conn, group_id, paper_id, added_via="manual")
    typer.echo(f"paper #{paper_id} añadido al grupo #{group_id}")


@group_app.command(name="expand-citations")
def group_expand_citations(
    group_id: int, limit: int = typer.Option(200, help="Máximo de citas a resolver")
):
    """Trae al corpus los papers citados por los del grupo (snowballing)."""
    conn = db.get_conn()
    r = groups.expand_citations(conn, group_id, limit=limit)
    typer.echo(
        f"pendientes: {r['pending']} · resueltos en OpenAlex: {r['fetched']} · "
        f"nuevos en la biblioteca: {r['inserted']} · ya existían: {r['duplicates']}"
    )


@group_app.command(name="synthesize")
def group_synthesize(
    group_id: int,
    topic: str = typer.Option(None, help="Tema de enfoque dentro del grupo"),
    limit: int = typer.Option(synthesize.DEFAULT_LIMIT, help="Máximo de papers a comparar"),
    full: bool = typer.Option(False, "--full", help="Todo el grupo por lotes (map-reduce)"),
):
    """Sintetiza los papers de un grupo (tendencias, contradicciones, huecos…)."""
    conn = db.get_conn()
    try:
        if full:
            s = synthesize.run_full(
                conn, topic=topic, batch_size=limit, progress=typer.echo, group_id=group_id
            )
        else:
            s = synthesize.run(conn, topic=topic, limit=limit, group_id=group_id)
    except (llm.OllamaError, ValueError) as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(1)
    typer.echo(f"Síntesis #{s.id} — {len(s.sources)} papers")
    _print_synthesis(s)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Interfaz de escucha"),
    port: int = typer.Option(8000),
    reload: bool = typer.Option(False, help="Autorecarga (desarrollo)"),
):
    """Levanta la web app (FastAPI)."""
    import uvicorn

    uvicorn.run("paperlab.web.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
