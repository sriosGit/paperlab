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

from .. import analyze, db, llm, synthesize
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


# ------------------------------------------------------------- biblioteca

@app.get("/", response_class=HTMLResponse)
def library(request: Request, q: str = "", source: str = "", year: str = "", status: str = ""):
    conn = db.get_conn()
    where, params = [], []
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
        "total": conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0],
        "sources": conn.execute("SELECT source, COUNT(*) n FROM papers GROUP BY source").fetchall(),
        "statuses": conn.execute("SELECT status, COUNT(*) n FROM papers GROUP BY status").fetchall(),
    }
    return templates.TemplateResponse(
        request, "library.html",
        {"papers": papers, "counts": counts, "q": q, "source": source,
         "year": year, "status": status, "job": job, "ollama_ok": llm.is_available()},
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
        {"p": paper, "summary": summary, "n_chunks": n_chunks},
    )


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
def create_synthesis(topic: str = Form(""), limit: int = Form(synthesize.DEFAULT_LIMIT)):
    topic = topic.strip() or None

    def _job():
        conn = db.get_conn()
        _log(f"sintetizando{f' «{topic}»' if topic else ''} (máx. {limit} papers)…")
        s = synthesize.run(conn, topic=topic, limit=limit)
        _log(f"síntesis #{s.id} lista: {len(s.sources)} papers comparados")

    if not _run_job("síntesis transversal", _job):
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


@app.post("/searches/{search_id}/run")
def run_saved_search(search_id: int):
    conn = db.get_conn()
    s = conn.execute("SELECT * FROM saved_searches WHERE id = ?", (search_id,)).fetchone()
    if not s:
        return RedirectResponse("/searches", status_code=303)
    result = run_search(conn, s["query"], s["sources"].split(","), s["max_results"])
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


# ------------------------------------------------------------- trabajos

@app.post("/jobs/process", response_class=HTMLResponse)
def start_process(request: Request):
    _run_job("procesar", _pipeline)
    return templates.TemplateResponse(request, "_job.html", {"job": job})


@app.post("/jobs/summarize", response_class=HTMLResponse)
def start_summarize(request: Request):
    _run_job("resumir", _summarize_all)
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
               WHERE papers_fts MATCH ? ORDER BY rank LIMIT ?""",
            (db.fts_escape(q), limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM papers ORDER BY added_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


@app.post("/api/ask")
def api_ask(payload: dict):
    conn = db.get_conn()
    answer = analyze.ask(conn, payload.get("question", ""))
    return {"answer": answer.text, "sources": answer.sources}
