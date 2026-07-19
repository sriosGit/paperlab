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
paperlab fetch-pdfs          # PDFs: URL guardada → arXiv → Unpaywall → Semantic Scholar
paperlab fetch-pdfs --retry  # reintenta todo paper sin PDF (re-indexa al conseguirlo)
paperlab process             # extrae texto, trocea, indexa FTS5 y calcula embeddings
paperlab summarize --limit 5 # resúmenes estructurados con el LLM
paperlab ask "¿qué métodos usan estos papers?"
paperlab review              # procedencia del corpus y papers por búsqueda
paperlab review --query "ia" --only   # los que SOLO trajo esa búsqueda (candidatos a descartar)
paperlab exclude 12,34 --reason "fuera de tema"   # descarta ruido (reversible con `include`)
paperlab sync-nas            # respalda los PDFs en el NAS (SRC Cloud) vía WebDAV
paperlab synthesize "protein folding"   # análisis transversal: compara papers entre sí
paperlab synthesize --full              # todo el corpus por lotes (map-reduce); tarda ~2 min por lote
paperlab synthesize --list              # síntesis guardadas; --show N para releer una
paperlab stats
```

## Web app

```sh
paperlab serve               # http://localhost:8000, escucha en 0.0.0.0
```

- **Biblioteca**: filtros por texto, fuente, año y estado; botones en background para procesar/resumir pendientes, enriquecer con OpenAlex, exportar a Obsidian y respaldar PDFs en el NAS. Muestra solo el corpus activo; los papers descartados viven en `/?excluded=1`.
- **Detalle de paper**: procedencia (qué búsquedas lo trajeron) y botón para excluirlo del corpus con un motivo.
- **Detalle de paper**: abstract, resumen estructurado generado bajo demanda.
- **Chat**: preguntas al corpus con RAG híbrido (embeddings + FTS5) y citas clicables.
- **Síntesis**: análisis transversal del corpus — compara los resúmenes de varios papers (por tema o los más recientes) y detecta tendencias, consensos, contradicciones, huecos abiertos, métodos transferibles y aplicaciones viables, todo citado con [n]. Con «todo el corpus (map-reduce)» analiza el corpus completo por lotes y combina los análisis parciales. Las síntesis quedan guardadas.
- **Búsquedas guardadas**: temas que se re-ejecutan con un clic (la ejecución automática programada llega en la iteración 2).
- API REST: `GET /api/papers?q=...` y `POST /api/ask {"question": "..."}`.

## Sin Docker

paperlab no usa contenedores: corre directo con un venv de Python y con Ollama
instalado de forma nativa en el host (`brew install ollama` o el instalador
oficial). No hay `Dockerfile` ni `docker-compose.yml` en el repo — si ves
contenedores corriendo en la máquina (p. ej. otros proyectos), no son parte de
este stack. `mise.toml` fija Python 3.12 como versión recomendada para el
venv (`mise install && mise use`), aunque cualquier 3.11+ funciona.

## Despliegue en la MacBook

1. Clona el repo e instala como arriba (con `mise install` si usas mise).
2. En `.env` deja `OLLAMA_BASE_URL=http://localhost:11434`.
3. `paperlab serve` — accesible en la tailnet vía `http://mb-2022:8000` (MagicDNS)
   o `http://mb-2022.tailad68d1.ts.net:8000` (FQDN completo), sin necesidad de IP.
4. Para exponerlo a internet con cloudflared:
   ```sh
   cloudflared tunnel --url http://localhost:8000
   ```
   Antes de compartirlo, protege el túnel con **Cloudflare Access** (la app no tiene login propio).

> Para desarrollar desde otra máquina de la tailnet usando el Ollama de la Mac,
> en la Mac ejecuta `launchctl setenv OLLAMA_HOST 0.0.0.0` y reinicia Ollama;
> en la otra máquina pon `OLLAMA_BASE_URL=http://mb-2022:11434` en `.env`.

## Calidad del corpus

Buscar términos ambiguos («IA», «agente») arrastra papers de otros campos
—*Rhizoctonia solani* **AG-1 IA**, supernovas **tipo Ia**— que contaminan
resúmenes, RAG y síntesis. Dos mecanismos lo contienen:

- **Exclusión**: `paperlab exclude <ids>` (o el botón en el detalle del paper)
  descarta un paper del corpus. No lo borra —así una búsqueda futura no lo
  vuelve a ingerir como nuevo— pero lo saca de resúmenes, chat, síntesis y
  exports. Reversible con `paperlab include`.
- **Procedencia**: cada paper registra qué búsqueda lo trajo (`paper_sources`).
  `paperlab review` resume el corpus por consulta y `--query X --only` lista los
  que *solo* entraron por esa consulta: los seguros de descartar en bloque si
  resultó ser ruido.

Las síntesis se auditan al mostrarlas: cobertura de citas, citas a fuentes
inexistentes y secciones sin ningún respaldo aparecen marcadas. La auditoría
verifica que las `[n]` existan, no que la afirmación se sostenga en el paper
citado — sigue haciendo falta abrir la fuente antes de apoyarse en un hallazgo.

## PDFs en el NAS (SRC Cloud)

paperlab respalda los PDFs en el NAS personal ([SRC Cloud](../src-cloud))
hablando con su **API WebDAV** (`/dav/...`, Basic auth) — no hace falta montar
nada. En `.env`:

```sh
NAS_BASE_URL=http://localhost:8000   # backend FastAPI del SRC Cloud
NAS_USERNAME=tu-usuario              # tu usuario del NAS
NAS_PASSWORD=tu-contraseña
NAS_PDF_DIR=paperlab/pdfs            # carpeta dentro de tu espacio del NAS
```

- `paperlab sync-nas` (o el botón «☁ PDFs al NAS» de la Biblioteca) sube los
  PDFs locales que falten en el NAS; es idempotente.
- `paperlab sync-nas --restore` además baja del NAS los que falten en disco
  (p. ej. tras clonar el repo en otra máquina).

El modelo es local-first: los PDFs se leen siempre del disco local y el NAS es
archivo/respaldo, así que `process` funciona aunque el NAS esté apagado. La
base SQLite se queda local (`PAPERLAB_DATA_DIR`) y los PDFs aparecen en el
explorador del SRC Cloud como archivos normales. Backup completo = copiar
`data/` (el NAS ya guarda los PDFs).

> Alternativa sin API: montar un share por SMB y apuntar `PAPERLAB_PDF_DIR`
> al punto de montaje (`paperlab relocate-pdfs` migra los ya descargados).

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
                  ├─ RAG: búsqueda híbrida (coseno numpy + FTS5, fusión RRF) → respuesta con citas
                  └─ synthesize.py → síntesis transversal (compara resúmenes entre papers)
web/ → FastAPI + Jinja2 + HTMX sobre la misma base de datos
```

Un solo archivo de datos (`data/paperlab.db`) + carpeta `data/pdfs/`. Backup = copiar `data/`.
