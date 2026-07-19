"""Configuración global, leída de variables de entorno / .env."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))

CONTACT_EMAIL = os.environ.get("CONTACT_EMAIL", "")

DATA_DIR = Path(os.environ.get("PAPERLAB_DATA_DIR", "data")).resolve()
# Los PDFs pueden vivir aparte (p. ej. en un NAS montado por SMB); la base
# SQLite se queda siempre en DATA_DIR local — WAL sobre red no es fiable.
_pdf_dir = os.environ.get("PAPERLAB_PDF_DIR")
PDF_DIR = Path(_pdf_dir).expanduser().resolve() if _pdf_dir else DATA_DIR / "pdfs"
DB_PATH = DATA_DIR / "paperlab.db"

USER_AGENT = f"paperlab/0.1 (mailto:{CONTACT_EMAIL})" if CONTACT_EMAIL else "paperlab/0.1"

_vault = os.environ.get("OBSIDIAN_VAULT_PATH")
OBSIDIAN_VAULT_PATH = Path(_vault).expanduser().resolve() if _vault else None


def ensure_dirs() -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
