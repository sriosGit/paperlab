"""Cliente Ollama (generación y embeddings).

La generación con qwen2.5:14b ocupa ~10 GB de RAM en la Mac de 16 GB, así que
se serializa con un lock global: un solo trabajo de generación a la vez aunque
lleguen varias peticiones concurrentes desde la web.
"""

import json
import threading

import httpx

from . import config

_generate_lock = threading.Lock()


class OllamaError(RuntimeError):
    pass


def is_available() -> bool:
    try:
        return httpx.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5).status_code == 200
    except httpx.HTTPError:
        return False


def generate(prompt: str, system: str | None = None, json_mode: bool = False) -> str:
    payload: dict = {
        "model": config.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_ctx": config.OLLAMA_NUM_CTX, "temperature": 0.2},
    }
    if system:
        payload["system"] = system
    if json_mode:
        payload["format"] = "json"
    with _generate_lock:
        try:
            resp = httpx.post(
                f"{config.OLLAMA_BASE_URL}/api/generate", json=payload, timeout=600
            )
            resp.raise_for_status()
        except httpx.HTTPError as e:
            raise OllamaError(f"Ollama no respondió ({config.OLLAMA_BASE_URL}): {e}") from e
    return resp.json().get("response", "")


def generate_json(prompt: str, system: str | None = None) -> dict:
    raw = generate(prompt, system=system, json_mode=True)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise OllamaError(f"el modelo no devolvió JSON válido: {raw[:200]}") from e


def embed(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    try:
        resp = httpx.post(
            f"{config.OLLAMA_BASE_URL}/api/embed",
            json={"model": config.OLLAMA_EMBED_MODEL, "input": texts},
            timeout=300,
        )
        resp.raise_for_status()
    except httpx.HTTPError as e:
        raise OllamaError(
            f"Error de embeddings ({config.OLLAMA_EMBED_MODEL}). "
            f"¿Hiciste 'ollama pull {config.OLLAMA_EMBED_MODEL}'? Detalle: {e}"
        ) from e
    return resp.json()["embeddings"]
