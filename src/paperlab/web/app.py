"""Web app de paperlab: biblioteca, chat RAG y búsquedas guardadas."""

import html
import json
import threading
from datetime import datetime, timezone
from pathlib import Path

import markdown as md
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .. import analyze, config, db, groups, llm, nas, query_builder, review, synthesize
from .. import pdf as pdf_mod
from ..ingest import run_search

app = FastAPI(title="paperlab")

_here = Path(__file__).parent
templates = Jinja2Templates(directory=_here / "templates")
templates.env.filters["fromjson"] = json.loads
# El texto viene del LLM y se renderiza con |safe: hay que escapar el HTML antes
# de convertir a markdown, o una respuesta con <script>/<img onerror> se ejecuta.
templates.env.filters["markdown"] = lambda t: md.markdown(html.escape(t or ""))
app.mount("/static", StaticFiles(directory=_here / "static"), name="static")

# ------------------------------------------------------------- trabajos en background

job = {"running": False, "name": "", "log": []}


def _log(msg: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    job["log"].append(f"{stamp} {msg}")
    job["log"] = job["log"][-30:]


def _run_job(name: str, fn) -> bool:
    """Lanza fn en un hilo si no hay otro trabajo corriendo."""
    if job["running"]:
        return False
    job.update(running=True, name=name, log=[])

    def wrapper():
        try:
            fn()
            _log("terminado ✓")
        except Exception as e:  # noqa: BLE001 — el hilo no debe morir en silencio
            _log(f"ERROR: {e}")
        finally:
            job["running"] = False

    threading.Thread(target=wrapper, daemon=True).start()
    return True


def _pipeline():
    conn = db.get_conn()
    _log("descargando PDFs pendientes…")
    _log(f"PDFs: {pdf_mod.fetch_pdfs(conn)}")
    _log("troceando e indexando…")
    _log(f"papers troceados: {analyze.ensure_chunks(conn)}")
    stale = analyze.stale_embeddings(conn)
    if stale:
        _log(f"↻ {stale} embeddings de otro modelo: se recalculan con {config.OLLAMA_EMBED_MODEL}")
    _log("calculando embeddings…")
    _log(f"chunks con embedding nuevo: {analyze.ensure_embeddings(conn)}")


def _summarize_all():
    conn = db.get_conn()
    ids = analyze.pending_summaries(conn)
    _log(f"papers por resumir: {len(ids)}")
    for i, pid in enumerate(ids, 1):
        try:
            analyze.summarize_paper(conn, pid)
            _log(f"[{i}/{len(ids)}] paper {pid} resumido")
        except Exception as e:  # noqa: BLE001
            _log(f"[{i}/{len(ids)}] paper {pid}: {e}")


def _enrich_openalex():
    from ..ingest import openalex
    from ..ingest.base import store_papers

    conn = db.get_conn()
    pending = conn.execute(
        """SELECT id, doi, arxiv_id FROM papers
           WHERE openalex_id IS NULL AND (doi IS NOT NULL OR arxiv_id IS NOT NULL)
             AND excluded = 0
           ORDER BY id"""
    ).fetchall()
    _log(f"papers por enriquecer: {len(pending)}")
    cites_before = conn.execute("SELECT COUNT(*) FROM citations").fetchone()[0]
    enriched = failed = 0
    for i, row in enumerate(pending, 1):
        try:
            paper = openalex.fetch_by_ids(row["doi"], row["arxiv_id"])
        except Exception as e:  # noqa: BLE001 — un paper no debe tumbar el lote
            _log(f"[{i}/{len(pending)}] paper {row['id']}: ERROR {e}")
            failed += 1
            continue
        if paper and paper.openalex_id:
            store_papers(conn, [paper])
            enriched += 1
        else:
            failed += 1
        _log(f"[{i}/{len(pending)}] paper {row['id']}: {'ok' if paper else 'sin match'}")
    cites_after = conn.execute("SELECT COUNT(*) FROM citations").fetchone()[0]
    _log(
        f"enriquecidos: {enriched} · sin match/error: {failed} · "
        f"citas nuevas: {cites_after - cites_before}"
    )


def _sync_nas():
    conn = db.get_conn()
    _log(f"sincronizando PDFs con el NAS ({config.NAS_BASE_URL})…")
    r = nas.sync_pdfs(conn, progress=_log)
    _log(
        f"subidos: {r['uploaded']} · ya en NAS: {r['already_on_nas']} · perdidos: {r['missing']}"
    )


def _export_obsidian():
    from ..export import obsidian

    if config.OBSIDIAN_VAULT_PATH is None:
        raise RuntimeError("define OBSIDIAN_VAULT_PATH en .env para exportar")
    conn = db.get_conn()
    _log(f"exportando a {config.OBSIDIAN_VAULT_PATH}…")
    s = obsidian.export_vault(conn, config.OBSIDIAN_VAULT_PATH)
    _log(
        f"papers: {s['notes']} · MOCs: {s['mocs']} · nuevos {s['created']}, "
        f"actualizados {s['updated']}, sin cambios {s['unchanged']}, "
        f"renombrados {s['renamed']} · huérfanas: {len(s['orphans'])}"
    )
    for h in s["orphans"]:
        _log(f"  huérfana (borra con el CLI --prune): {h}")


# ------------------------------------------------------------- biblioteca

@app.get("/", response_class=HTMLResponse)
def library(
    request: Request, q: str = "", source: str = "", year: str = "",
    status: str = "", excluded: str = "",
):
    conn = db.get_conn()
    # por defecto la biblioteca muestra el corpus activo; ?excluded=1 la papelera
    where, params = ["p.excluded = ?"], [1 if excluded else 0]
    if q:
        ids = [
            r["rowid"]
            for r in conn.execute(
                "SELECT rowid FROM papers_fts WHERE papers_fts MATCH ? ORDER BY rank LIMIT 500",
                (db.fts_escape(q),),
            )
        ]
        where.append(f"p.id IN ({','.join('?' * len(ids)) or 'NULL'})")
        params.extend(ids)
    if source:
        where.append("p.source = ?")
        params.append(source)
    if year:
        where.append("p.year = ?")
        params.append(int(year))
    if status:
        where.append("p.status = ?")
        params.append(status)
    sql = f"""SELECT p.*, (s.paper_id IS NOT NULL) AS has_summary
              FROM papers p LEFT JOIN summaries s ON s.paper_id = p.id
              {"WHERE " + " AND ".join(where) if where else ""}
              ORDER BY p.added_at DESC, p.id DESC LIMIT 200"""
    papers = conn.execute(sql, params).fetchall()
    counts = {
        **review.stats(conn),
        "sources": conn.execute(
            "SELECT source, COUNT(*) n FROM papers WHERE excluded = 0 GROUP BY source"
        ).fetchall(),
        "statuses": conn.execute(
            "SELECT status, COUNT(*) n FROM papers WHERE excluded = 0 GROUP BY status"
        ).fetchall(),
    }
    return templates.TemplateResponse(
        request, "library.html",
        {"papers": papers, "counts": counts, "q": q, "source": source,
         "year": year, "status": status, "excluded": excluded, "job": job,
         "ollama_ok": llm.is_available(),
         "obsidian_vault": config.OBSIDIAN_VAULT_PATH, "nas_ok": nas.enabled()},
    )


@app.get("/paper/{paper_id}", response_class=HTMLResponse)
def paper_detail(request: Request, paper_id: int):
    conn = db.get_conn()
    paper = conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
    if not paper:
        return RedirectResponse("/", status_code=303)
    summary = conn.execute("SELECT * FROM summaries WHERE paper_id = ?", (paper_id,)).fetchone()
    n_chunks = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE paper_id = ?", (paper_id,)
    ).fetchone()[0]
    return templates.TemplateResponse(
        request, "paper.html",
        {"p": paper, "summary": summary, "n_chunks": n_chunks,
         "provenance": review.provenance(conn, paper_id)},
    )


@app.post("/paper/{paper_id}/exclude")
def paper_exclude(paper_id: int, reason: str = Form("")):
    conn = db.get_conn()
    review.exclude(conn, [paper_id], reason)
    return RedirectResponse(f"/paper/{paper_id}", status_code=303)


@app.post("/paper/{paper_id}/include")
def paper_include(paper_id: int):
    conn = db.get_conn()
    review.include(conn, [paper_id])
    return RedirectResponse(f"/paper/{paper_id}", status_code=303)


@app.post("/paper/{paper_id}/summarize", response_class=HTMLResponse)
def paper_summarize(request: Request, paper_id: int):
    conn = db.get_conn()
    try:
        analyze.summarize_paper(conn, paper_id)
        error = None
    except (llm.OllamaError, ValueError) as e:
        error = str(e)
    summary = conn.execute("SELECT * FROM summaries WHERE paper_id = ?", (paper_id,)).fetchone()
    return templates.TemplateResponse(
        request, "_summary.html", {"summary": summary, "error": error, "p": {"id": paper_id}}
    )


# ------------------------------------------------------------- chat RAG

@app.get("/chat", response_class=HTMLResponse)
def chat(request: Request):
    return templates.TemplateResponse(request, "chat.html", {"ollama_ok": llm.is_available()})


@app.post("/chat/ask", response_class=HTMLResponse)
def chat_ask(request: Request, question: str = Form(...)):
    conn = db.get_conn()
    try:
        answer = analyze.ask(conn, question)
        error = None
    except llm.OllamaError as e:
        answer, error = None, str(e)
    return templates.TemplateResponse(
        request, "_answer.html", {"question": question, "answer": answer, "error": error}
    )


# ------------------------------------------------------------- síntesis transversal

@app.get("/synthesis", response_class=HTMLResponse)
def syntheses_list(request: Request, error: str = ""):
    conn = db.get_conn()
    rows = [
        {**dict(r), "n_papers": len(json.loads(r["paper_ids"]))}
        for r in synthesize.list_all(conn)
    ]
    n_summaries = conn.execute("SELECT COUNT(*) FROM summaries").fetchone()[0]
    return templates.TemplateResponse(
        request, "syntheses.html",
        {"syntheses": rows, "n_summaries": n_summaries, "job": job,
         "error": error, "ollama_ok": llm.is_available()},
    )


@app.post("/synthesis")
def create_synthesis(
    topic: str = Form(""), limit: int = Form(synthesize.DEFAULT_LIMIT), full: str = Form(""),
):
    topic = topic.strip() or None

    def _job():
        conn = db.get_conn()
        if full:
            _log(f"map-reduce del corpus{f' «{topic}»' if topic else ''} (lotes de {limit})…")
            s = synthesize.run_full(conn, topic=topic, batch_size=limit, progress=_log)
        else:
            _log(f"sintetizando{f' «{topic}»' if topic else ''} (máx. {limit} papers)…")
            s = synthesize.run(conn, topic=topic, limit=limit)
        _log(f"síntesis #{s.id} lista: {len(s.sources)} papers comparados")

    name = "síntesis transversal" + (" (corpus completo)" if full else "")
    if not _run_job(name, _job):
        return RedirectResponse("/synthesis?error=ya+hay+un+trabajo+en+curso", status_code=303)
    return RedirectResponse("/synthesis", status_code=303)


@app.get("/synthesis/{synthesis_id}", response_class=HTMLResponse)
def synthesis_detail(request: Request, synthesis_id: int):
    conn = db.get_conn()
    s = synthesize.get(conn, synthesis_id)
    if not s:
        return RedirectResponse("/synthesis", status_code=303)
    return templates.TemplateResponse(
        request, "synthesis.html", {"s": s, "section_labels": synthesize.SECTIONS}
    )


@app.post("/synthesis/{synthesis_id}/delete")
def delete_synthesis(synthesis_id: int):
    conn = db.get_conn()
    conn.execute("DELETE FROM syntheses WHERE id = ?", (synthesis_id,))
    conn.commit()
    return RedirectResponse("/synthesis", status_code=303)


# ------------------------------------------------------------- búsquedas guardadas

@app.get("/searches", response_class=HTMLResponse)
def searches(request: Request, msg: str = ""):
    conn = db.get_conn()
    rows = conn.execute("SELECT * FROM saved_searches ORDER BY id").fetchall()
    return templates.TemplateResponse(request, "searches.html", {"searches": rows, "msg": msg})


@app.post("/searches")
def create_search(
    name: str = Form(...), query: str = Form(...),
    sources: str = Form("arxiv,openalex"), max_results: int = Form(50),
):
    conn = db.get_conn()
    conn.execute(
        "INSERT INTO saved_searches (name, query, sources, max_results) VALUES (?, ?, ?, ?)",
        (name, query, sources, max_results),
    )
    conn.commit()
    return RedirectResponse("/searches", status_code=303)


@app.post("/searches/ai/suggest", response_class=HTMLResponse)
def searches_ai_suggest(request: Request, ask: str = Form(...)):
    try:
        plan, error = query_builder.suggest(ask), None
    except llm.OllamaError as e:
        plan, error = None, str(e)
    return templates.TemplateResponse(
        request, "_ai_preview.html", {"plan": plan, "error": error}
    )


@app.post("/searches/ai/run")
def searches_ai_run(
    query: str = Form(...),
    sources: list[str] = Form(...),
    from_year: str = Form(""),
    to_year: str = Form(""),
    limit: int = Form(50),
):
    conn = db.get_conn()
    result = run_search(
        conn, query, sources, limit,
        from_year=int(from_year) if from_year else None,
        to_year=int(to_year) if to_year else None,
    )
    msg = "; ".join(f"{src}: {info}" for src, info in result.items())
    return RedirectResponse(f"/searches?msg={msg}", status_code=303)


@app.post("/searches/ai/save")
def searches_ai_save(
    query: str = Form(...),
    sources: list[str] = Form(...),
    from_year: str = Form(""),
    to_year: str = Form(""),
    limit: int = Form(50),
    name: str = Form(""),
):
    conn = db.get_conn()
    conn.execute(
        "INSERT INTO saved_searches (name, query, sources, max_results, from_year, to_year) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            name.strip() or query,
            query,
            ",".join(sources),
            limit,
            int(from_year) if from_year else None,
            int(to_year) if to_year else None,
        ),
    )
    conn.commit()
    return RedirectResponse("/searches", status_code=303)


@app.post("/searches/{search_id}/run")
def run_saved_search(search_id: int):
    conn = db.get_conn()
    s = conn.execute("SELECT * FROM saved_searches WHERE id = ?", (search_id,)).fetchone()
    if not s:
        return RedirectResponse("/searches", status_code=303)
    result = run_search(
        conn,
        s["query"],
        s["sources"].split(","),
        s["max_results"],
        search_id=search_id,
        from_year=s["from_year"],
        to_year=s["to_year"],
    )
    conn.execute(
        "UPDATE saved_searches SET last_run_at = datetime('now') WHERE id = ?", (search_id,)
    )
    conn.commit()
    msg = "; ".join(f"{src}: {info}" for src, info in result.items())
    return RedirectResponse(f"/searches?msg={msg}", status_code=303)


@app.post("/searches/{search_id}/delete")
def delete_search(search_id: int):
    conn = db.get_conn()
    conn.execute("DELETE FROM saved_searches WHERE id = ?", (search_id,))
    conn.commit()
    return RedirectResponse("/searches", status_code=303)


# ------------------------------------------------------------- grupos

@app.get("/groups", response_class=HTMLResponse)
def groups_list(request: Request):
    conn = db.get_conn()
    return templates.TemplateResponse(
        request, "groups.html", {"groups": groups.list_all(conn)}
    )


@app.post("/groups")
def groups_create(name: str = Form(...), description: str = Form("")):
    conn = db.get_conn()
    gid = groups.create(conn, name, description)
    return RedirectResponse(f"/groups/{gid}", status_code=303)


@app.get("/groups/{group_id}", response_class=HTMLResponse)
def group_detail(request: Request, group_id: int):
    conn = db.get_conn()
    g = groups.get(conn, group_id)
    if not g:
        return RedirectResponse("/groups", status_code=303)
    linked = groups.linked_searches(conn, group_id)
    linked_ids = {r["id"] for r in linked}
    available_searches = [
        r for r in conn.execute("SELECT * FROM saved_searches ORDER BY name").fetchall()
        if r["id"] not in linked_ids
    ]
    syntheses = synthesize.list_for_group(conn, group_id)
    contradicted: set[int] = set()
    if syntheses:
        latest = synthesize.get(conn, syntheses[0]["id"])
        if latest:
            contradicted = synthesize.contradicted_papers(latest)
    return templates.TemplateResponse(
        request, "group.html",
        {"g": g, "linked_searches": linked, "available_searches": available_searches,
         "members": groups.members(conn, group_id), "contradicted": contradicted,
         "syntheses": [
             {**dict(r), "n_papers": len(json.loads(r["paper_ids"]))} for r in syntheses
         ],
         "job": job, "ollama_ok": llm.is_available()},
    )


@app.post("/groups/{group_id}/delete")
def group_delete(group_id: int):
    groups.delete(db.get_conn(), group_id)
    return RedirectResponse("/groups", status_code=303)


@app.post("/groups/{group_id}/searches")
def group_link_search(group_id: int, search_id: int = Form(...)):
    groups.link_search(db.get_conn(), group_id, search_id)
    return RedirectResponse(f"/groups/{group_id}", status_code=303)


@app.post("/groups/{group_id}/searches/{search_id}/unlink")
def group_unlink_search(group_id: int, search_id: int):
    groups.unlink_search(db.get_conn(), group_id, search_id)
    return RedirectResponse(f"/groups/{group_id}", status_code=303)


@app.post("/groups/{group_id}/searches/{search_id}/run")
def group_run_search(group_id: int, search_id: int):
    conn = db.get_conn()
    s = conn.execute("SELECT * FROM saved_searches WHERE id = ?", (search_id,)).fetchone()
    if s:
        run_search(
            conn, s["query"], s["sources"].split(","), s["max_results"],
            search_id=search_id, from_year=s["from_year"], to_year=s["to_year"],
        )
        conn.execute(
            "UPDATE saved_searches SET last_run_at = datetime('now') WHERE id = ?", (search_id,)
        )
        conn.commit()
        groups.sync_search(conn, search_id)
    return RedirectResponse(f"/groups/{group_id}", status_code=303)


@app.post("/groups/{group_id}/papers")
def group_add_paper(group_id: int, paper_id: int = Form(...)):
    conn = db.get_conn()
    if conn.execute("SELECT 1 FROM papers WHERE id = ?", (paper_id,)).fetchone():
        groups.add_paper(conn, group_id, paper_id, added_via="manual")
    return RedirectResponse(f"/groups/{group_id}", status_code=303)


@app.post("/groups/{group_id}/papers/{paper_id}/remove")
def group_remove_paper(group_id: int, paper_id: int):
    groups.remove_paper(db.get_conn(), group_id, paper_id)
    return RedirectResponse(f"/groups/{group_id}", status_code=303)


def _expand_citations(group_id: int):
    conn = db.get_conn()
    _log(f"buscando referencias citadas por el grupo #{group_id}…")
    r = groups.expand_citations(conn, group_id)
    _log(
        f"pendientes: {r['pending']} · resueltos en OpenAlex: {r['fetched']} · "
        f"nuevos en la biblioteca: {r['inserted']} · ya existían: {r['duplicates']}"
    )


def _group_synthesize(group_id: int, topic: str | None, limit: int, full: bool):
    conn = db.get_conn()
    if full:
        _log(f"map-reduce del grupo #{group_id} (lotes de {limit})…")
        s = synthesize.run_full(conn, topic=topic, batch_size=limit, progress=_log, group_id=group_id)
    else:
        _log(f"sintetizando grupo #{group_id} (máx. {limit} papers)…")
        s = synthesize.run(conn, topic=topic, limit=limit, group_id=group_id)
    _log(f"síntesis #{s.id} lista: {len(s.sources)} papers comparados")


@app.post("/groups/{group_id}/expand-citations", response_class=HTMLResponse)
def group_start_expand_citations(request: Request, group_id: int):
    _run_job("buscar referencias citadas", lambda: _expand_citations(group_id))
    return templates.TemplateResponse(request, "_job.html", {"job": job})


@app.post("/groups/{group_id}/synthesize", response_class=HTMLResponse)
def group_start_synthesize(
    request: Request, group_id: int,
    topic: str = Form(""), limit: int = Form(synthesize.DEFAULT_LIMIT), full: str = Form(""),
):
    _run_job(
        "sintetizar grupo",
        lambda: _group_synthesize(group_id, topic.strip() or None, limit, bool(full)),
    )
    return templates.TemplateResponse(request, "_job.html", {"job": job})


# ------------------------------------------------------------- trabajos

@app.post("/jobs/process", response_class=HTMLResponse)
def start_process(request: Request):
    _run_job("procesar", _pipeline)
    return templates.TemplateResponse(request, "_job.html", {"job": job})


@app.post("/jobs/summarize", response_class=HTMLResponse)
def start_summarize(request: Request):
    _run_job("resumir", _summarize_all)
    return templates.TemplateResponse(request, "_job.html", {"job": job})


@app.post("/jobs/enrich", response_class=HTMLResponse)
def start_enrich(request: Request):
    _run_job("enriquecer OpenAlex", _enrich_openalex)
    return templates.TemplateResponse(request, "_job.html", {"job": job})


@app.post("/jobs/export-obsidian", response_class=HTMLResponse)
def start_export_obsidian(request: Request):
    _run_job("exportar a Obsidian", _export_obsidian)
    return templates.TemplateResponse(request, "_job.html", {"job": job})


@app.post("/jobs/sync-nas", response_class=HTMLResponse)
def start_sync_nas(request: Request):
    _run_job("respaldar PDFs en el NAS", _sync_nas)
    return templates.TemplateResponse(request, "_job.html", {"job": job})


@app.get("/jobs/status", response_class=HTMLResponse)
def job_status(request: Request):
    return templates.TemplateResponse(request, "_job.html", {"job": job})


# ------------------------------------------------------------- API REST

@app.get("/api/papers")
def api_papers(q: str = "", limit: int = 100):
    conn = db.get_conn()
    if q:
        rows = conn.execute(
            """SELECT p.* FROM papers_fts f JOIN papers p ON p.id = f.rowid
               WHERE papers_fts MATCH ? AND p.excluded = 0 ORDER BY rank LIMIT ?""",
            (db.fts_escape(q), limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM papers WHERE excluded = 0 ORDER BY added_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


@app.post("/api/ask")
def api_ask(payload: dict):
    conn = db.get_conn()
    answer = analyze.ask(conn, payload.get("question", ""))
    return {"answer": answer.text, "sources": answer.sources}
