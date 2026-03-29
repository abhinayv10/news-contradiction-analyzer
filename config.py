"""Runtime configuration from environment (no hardcoded provider)."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _get(key: str, default: str | None = None) -> str | None:
    v = os.environ.get(key, default)
    return v if v not in ("", None) else default


def _truthy(key: str, default: str = "false") -> bool:
    return str(_get(key, default) or default).lower() in ("1", "true", "yes", "on")


LLM_API_KEY: str | None = _get("LLM_API_KEY")
LLM_BASE_URL: str = _get("LLM_BASE_URL", "https://api.openai.com/v1") or "https://api.openai.com/v1"
LLM_MODEL: str = _get("LLM_MODEL", "gpt-4o-mini") or "gpt-4o-mini"
LLM_TEMPERATURE: float = float(_get("LLM_TEMPERATURE", "0") or "0")
LLM_EMBEDDING_MODEL: str | None = _get("LLM_EMBEDDING_MODEL")

# Demo/showcase: use heuristic extraction + rule-based classification (fast, offline).
DEMO_MODE: bool = _truthy("DEMO_MODE", "false")

# Chunk size for long articles (characters, conservative for context limits)
CLAIM_CHUNK_CHARS: int = int(_get("CLAIM_CHUNK_CHARS", "6000") or "6000")

# Minimum rapidfuzz score (0-100) to consider a pair aligned
ALIGNMENT_MIN_SCORE: float = float(_get("ALIGNMENT_MIN_SCORE", "55") or "55")

# Numeric comparison tolerances for rule-based labeling
AMOUNT_REL_TOLERANCE: float = float(_get("AMOUNT_REL_TOLERANCE", "0.03") or "0.03")  # 3% relative
PERCENT_ABS_TOLERANCE: float = float(_get("PERCENT_ABS_TOLERANCE", "1.0") or "1.0")  # 1 percentage point
QUOTE_AGREE_THRESHOLD: float = float(_get("QUOTE_AGREE_THRESHOLD", "90") or "90")
QUOTE_CONFLICT_THRESHOLD: float = float(_get("QUOTE_CONFLICT_THRESHOLD", "60") or "60")

CACHE_DIR: Path | None = None
_cd = _get("CACHE_DIR", ".cache")
if _cd:
    CACHE_DIR = Path(_cd)
