"""CLI de paperlab (typer)."""

import json

import typer

from . import analyze, db, llm
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


@app.command(name="fetch-pdfs")
def fetch_pdfs(limit: int = typer.Option(None, help="Máximo de PDFs a descargar")):
    """Descarga los PDFs de los papers pendientes (arXiv directo, Unpaywall por DOI)."""
    conn = db.get_conn()
    typer.echo(json.dumps(pdf_mod.fetch_pdfs(conn, limit), ensure_ascii=False))


@app.command()
def process(no_embed: bool = typer.Option(False, "--no-embed", help="Solo trocear e indexar FTS, sin embeddings")):
    """Extrae texto, trocea, indexa (FTS5) y calcula embeddings pendientes."""
    conn = db.get_conn()
    n = analyze.ensure_chunks(conn)
    typer.echo(f"papers troceados: {n}")
    if no_embed:
        return
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
    typer.echo(f"papers: {total}  " + "  ".join(f"{r[0]}={r[1]}" for r in by_status))
    typer.echo(f"chunks: {chunks} (con embedding: {embedded})")
    typer.echo(f"resúmenes: {summaries}")
    typer.echo(f"ollama disponible: {'sí' if llm.is_available() else 'NO'}")


@app.command(name="enrich-openalex")
def enrich_openalex(limit: int = typer.Option(None, help="Máximo de papers a enriquecer")):
    """Rellena openalex_id y citas de papers que solo tienen DOI/arXiv (densifica el grafo)."""
    from .ingest import openalex
    from .ingest.base import store_papers

    conn = db.get_conn()
    pendientes = conn.execute(
        """SELECT id, doi, arxiv_id FROM papers
           WHERE openalex_id IS NULL AND (doi IS NOT NULL OR arxiv_id IS NOT NULL)
           ORDER BY id"""
    ).fetchall()
    if limit:
        pendientes = pendientes[:limit]
    if not pendientes:
        typer.echo("no hay papers con DOI/arXiv pendientes de enriquecer")
        return

    citas_antes = conn.execute("SELECT COUNT(*) FROM citations").fetchone()[0]
    enriquecidos = fallidos = 0
    for i, row in enumerate(pendientes, 1):
        try:
            paper = openalex.fetch_by_ids(row["doi"], row["arxiv_id"])
        except Exception as exc:  # noqa: BLE001 — un paper no debe tumbar el lote
            typer.echo(f"[{i}/{len(pendientes)}] paper {row['id']}: ERROR {exc}", err=True)
            fallidos += 1
            continue
        if paper and paper.openalex_id:
            store_papers(conn, [paper])
            enriquecidos += 1
        else:
            fallidos += 1
        typer.echo(f"[{i}/{len(pendientes)}] paper {row['id']}: {'ok' if paper else 'sin match'}")

    citas_despues = conn.execute("SELECT COUNT(*) FROM citations").fetchone()[0]
    typer.echo(
        f"enriquecidos: {enriquecidos} · sin match/error: {fallidos} · "
        f"citas nuevas: {citas_despues - citas_antes}"
    )


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
