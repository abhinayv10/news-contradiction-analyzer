"""Highlight text differences between two strings using difflib (optional UI aid)."""

from __future__ import annotations

import difflib


def unified_diff_snippet(a: str, b: str, width: int = 40) -> str:
    """Short unified diff for two value strings (truncated)."""
    a = (a or "")[:500]
    b = (b or "")[:500]
    lines = list(
        difflib.unified_diff(
            a.splitlines(),
            b.splitlines(),
            lineterm="",
            n=1,
        )
    )
    out = "\n".join(lines[: width // 2])
    return out if out else "(no diff)"


def ratio(a: str, b: str) -> float:
    """SequenceMatcher ratio 0-1 for side-by-side similarity."""
    return difflib.SequenceMatcher(None, a or "", b or "").ratio()
