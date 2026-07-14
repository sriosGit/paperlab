# paperlab

Recopilador y analizador **local-first** de artículos científicos. Busca en arXiv y OpenAlex, descarga PDFs open-access, los indexa (texto completo + embeddings) y los analiza con un LLM local vía Ollama: resúmenes estructurados y chat RAG con citas. Todo corre en tu máquina, costo $0.

## Requisitos

- Python 3.11+ (hay `mise.toml` para [mise](https://mise.jdx.dev))
- [Ollama](https://ollama.com) con los modelos:
  ```sh
  ollama pull qwen2.5:14b        # generación (ya lo tienes)
  ollama pull nomic-embed-text   # embeddings (~274 MB)
  ```

## Instalación

```sh
git clone <este-repo> && cd paperlab
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env   # edita CONTACT_EMAIL (OpenAlex/Unpaywall lo piden)
```

## Uso (CLI)

```sh
paperlab search "protein structure prediction" --source arxiv,openalex --limit 30
paperlab fetch-pdfs          # descarga PDFs (arXiv directo; Unpaywall por DOI)
paperlab process             # extrae texto, trocea, indexa FTS5 y calcula embeddings
paperlab summarize --limit 5 # resúmenes estructurados con el LLM
paperlab ask "¿qué métodos usan estos papers?"
paperlab stats
```

## Web app

```sh
paperlab serve               # http://localhost:8000, escucha en 0.0.0.0
```

- **Biblioteca**: filtros por texto, fuente, año y estado; botones para procesar/resumir pendientes en background.
- **Detalle de paper**: abstract, resumen estructurado generado bajo demanda.
- **Chat**: preguntas al corpus con RAG híbrido (embeddings + FTS5) y citas clicables.
- **Búsquedas guardadas**: temas que se re-ejecutan con un clic (la ejecución automática programada llega en la iteración 2).
- API REST: `GET /api/papers?q=...` y `POST /api/ask {"question": "..."}`.

## Despliegue en la MacBook

1. Clona el repo e instala como arriba (con `mise install` si usas mise).
2. En `.env` deja `OLLAMA_BASE_URL=http://localhost:11434`.
3. `paperlab serve` — accesible en la tailnet vía `http://100.83.237.84:8000`.
4. Para exponerlo a internet con cloudflared:
   ```sh
   cloudflared tunnel --url http://localhost:8000
   ```
   Antes de compartirlo, protege el túnel con **Cloudflare Access** (la app no tiene login propio).

> Para desarrollar desde otra máquina de la tailnet usando el Ollama de la Mac,
> en la Mac ejecuta `launchctl setenv OLLAMA_HOST 0.0.0.0` y reinicia Ollama;
> en la otra máquina pon `OLLAMA_BASE_URL=http://100.83.237.84:11434` en `.env`.

## Notas de recursos (Mac de 16 GB)

- `qwen2.5:14b` ocupa ~10 GB al cargarse: la app serializa la generación (un trabajo a la vez).
- Lotes grandes (100+ papers) conviene lanzarlos de noche: `paperlab summarize` o el botón «Resumir pendientes».
- El contexto de generación por defecto es 8192 tokens (`OLLAMA_NUM_CTX`); súbelo con cuidado, el KV cache consume RAM.

## Arquitectura

```
fuentes (arXiv, OpenAlex, Unpaywall)
   └─ ingest/ → SQLite (papers, dedup por DOI/arXiv/OpenAlex/título)
        └─ pdf.py → PDFs → texto (PyMuPDF) → chunks (~800 tokens)
             └─ analyze.py → embeddings (Ollama) como BLOB + índice FTS5
                  ├─ resúmenes estructurados (JSON) por paper
                  └─ RAG: búsqueda híbrida (coseno numpy + FTS5, fusión RRF) → respuesta con citas
web/ → FastAPI + Jinja2 + HTMX sobre la misma base de datos
```

Un solo archivo de datos (`data/paperlab.db`) + carpeta `data/pdfs/`. Backup = copiar `data/`.
