"""
Configurable OpenAI-compatible LLM client (chat + optional embeddings).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from openai import OpenAI

import config


def _cache_key(prefix: str, payload: str) -> str:
    h = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
    return f"{prefix}_{h}.json"


def _read_cache(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write_cache(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def get_client() -> OpenAI:
    if not config.LLM_API_KEY:
        raise RuntimeError("LLM_API_KEY is not set. Copy .env.example to .env and configure.")
    return OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)


def chat_json(
    system: str,
    user: str,
    *,
    cache_prefix: str = "chat",
    use_cache: bool = True,
) -> str:
    """
    Complete chat and return assistant message content (expected JSON string).
    Optional disk cache when CACHE_DIR is set.
    """
    payload = system + "\n" + user
    cache_path: Path | None = None
    if use_cache and config.CACHE_DIR:
        cache_path = config.CACHE_DIR / _cache_key(cache_prefix, payload)
        cached = _read_cache(cache_path)
        if isinstance(cached, dict) and "content" in cached:
            return str(cached["content"])

    client = get_client()
    resp = client.chat.completions.create(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    # Some models ignore array request; we use json_object wrapper in extractors when needed
    content = resp.choices[0].message.content or "{}"
    if cache_path:
        _write_cache(cache_path, {"content": content})
    return content


def chat_json_array_wrapper(
    system: str,
    user: str,
    *,
    cache_prefix: str = "claims",
) -> str:
    """JSON object mode: system prompt must require a top-level object (e.g. {\"claims\": [...]})."""
    payload = system + "\n" + user
    cache_path: Path | None = None
    if config.CACHE_DIR:
        cache_path = config.CACHE_DIR / _cache_key(cache_prefix, payload)
        cached = _read_cache(cache_path)
        if isinstance(cached, dict) and "content" in cached:
            return str(cached["content"])

    client = get_client()
    resp = client.chat.completions.create(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or '{"claims":[]}'
    if cache_path:
        _write_cache(cache_path, {"content": content})
    return content


def embed_texts(texts: list[str]) -> list[list[float]] | None:
    """Return embeddings if LLM_EMBEDDING_MODEL is configured; else None."""
    if not config.LLM_EMBEDDING_MODEL or not texts:
        return None
    client = get_client()
    resp = client.embeddings.create(model=config.LLM_EMBEDDING_MODEL, input=texts)
    ordered = sorted(resp.data, key=lambda d: d.index)
    return [d.embedding for d in ordered]
