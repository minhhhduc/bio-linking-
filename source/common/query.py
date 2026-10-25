"""Dense query shared by the baseline and GRF.

The mention stays **in place** in its sentence. The sentence is returned as
written, with ``[SEP]`` markers inserted immediately before and after the
mention, and the encoder pools only the token states between the markers (see
``SentenceTransformerEncoder.encode_mention_in_context``): the words before and
after the mention stay context, never averaged into the vector.
"""

from __future__ import annotations

import re

SEPARATOR = " [SEP] "
_WHITESPACE = re.compile(r"\S+")


def _locate(
    context: str,
    mention: str,
    start_token: int | None,
    end_token: int | None,
) -> tuple[int, int]:
    """Character offsets of ``mention`` in ``context``.

    Gold whitespace-token indices (inclusive, 0-based) win when usable; the
    mention string is located otherwise.
    """
    if (
        start_token is not None
        and end_token is not None
        and 0 <= start_token <= end_token
    ):
        tokens = list(_WHITESPACE.finditer(context))
        if end_token < len(tokens):
            return tokens[start_token].start(), tokens[end_token].end()

    position = context.find(mention)
    if position >= 0:
        return position, position + len(mention)
    raise ValueError(
        f"mention {mention!r} does not occur in context and the token span "
        f"({start_token!r}, {end_token!r}) is not usable"
    )


def mention_query(
    mention: str,
    context: str,
    *,
    start_token: int | None = None,
    end_token: int | None = None,
) -> tuple[str, int, int]:
    """Return ``(text, start, end)``: the marked sentence and the mention span.

    ``text`` is ``context`` with ``[SEP]`` around the mention, so the words
    before the mention remain before it. ``[start, end)`` are character offsets
    of the mention inside ``text``.
    """
    if not isinstance(mention, str) or not mention.strip():
        raise ValueError("mention must be a non-empty string")
    if not isinstance(context, str) or not context.strip():
        raise ValueError("context must be a non-empty string")

    start, end = _locate(context, mention, start_token, end_token)
    text = f"{context[:start]}{SEPARATOR}{mention}{SEPARATOR}{context[end:]}"
    mention_start = start + len(SEPARATOR)
    return text, mention_start, mention_start + len(mention)
