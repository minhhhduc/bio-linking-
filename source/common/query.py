"""Query text embedded for dense retrieval, shared by the baseline and GRF.

Both methods embed the same string — the mention plus the sentence it occurs
in — so their retrieval numbers are comparable.
"""

from __future__ import annotations

SEPARATOR = " [SEP] "


def query_text(mention: str, context: str) -> str:
    if not isinstance(mention, str) or not mention.strip():
        raise ValueError("mention must be a non-empty string")
    if not isinstance(context, str) or not context.strip():
        raise ValueError("context must be a non-empty string")
    return f"{mention}{SEPARATOR}{context}"
