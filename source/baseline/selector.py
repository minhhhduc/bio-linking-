"""Listwise LLM selection over retrieved candidates (baseline stage 2).

The selector only sees the mention, its context and the candidate list saved by
the retrieval stage. It returns a full re-ranking of those candidates, so the
end-to-end stage can report acc@1 and acc@5.
"""

from __future__ import annotations

import json
import re
from typing import Mapping, Sequence

from common.schemas import Candidate

PROMPT_TEMPLATE = """You are a biomedical entity linker.
Mention: {mention}
Context: {context}

Candidates:
{rows}

Rank ALL candidates from the most to the least likely concept that the mention
refers to in this context. Use the CUI exactly as written.

Return JSON only: {{"ranking": ["<cui>", "<cui>", ...]}}"""


class SelectionError(RuntimeError):
    """The answer could not be mapped onto the candidate list."""


def _cui_pattern(cui: str) -> re.Pattern[str]:
    # A CUI must stand alone: not glued to an alphanumeric run, and not the
    # suffix of another namespaced id (e.g. "0001" inside "CUI:0001").
    return re.compile(r"(?<![A-Za-z0-9_:])" + re.escape(cui) + r"(?![A-Za-z0-9_])")


def build_prompt(
    mention: str,
    context: str,
    candidates: Sequence[Candidate],
    concept_names: Mapping[str, str] | None = None,
) -> str:
    names = concept_names or {}
    rows = "\n".join(
        f"{index}. CUI: {candidate.cui} | "
        f"Name: {names.get(candidate.cui) or candidate.surface} | "
        f"Surface: {candidate.surface}"
        for index, candidate in enumerate(candidates, start=1)
    )
    return PROMPT_TEMPLATE.format(mention=mention, context=context, rows=rows)


def _from_json(answer: str, cuis: list[str]) -> list[str] | None:
    match = re.search(r"\{.*\}", answer, re.DOTALL)
    if match is None:
        return None
    try:
        payload = json.loads(match.group(0))
    except (json.JSONDecodeError, TypeError):
        return None
    ranking = payload.get("ranking") if isinstance(payload, dict) else None
    if not isinstance(ranking, list):
        return None
    known = set(cuis)
    return [cui for cui in ranking if isinstance(cui, str) and cui in known]


def _from_mentions(answer: str, cuis: list[str]) -> list[str]:
    found: list[tuple[int, str]] = []
    for cui in cuis:
        for match in _cui_pattern(cui).finditer(answer):
            found.append((match.start(), cui))
    found.sort()
    return [cui for _position, cui in found]


def parse_ranking(answer: str, cuis: Sequence[str]) -> list[str]:
    """Map the model answer onto ``cuis``, in the order the model gave.

    Every candidate appears exactly once: whatever the model leaves out is
    appended in retrieval order, so the result is still a full ranking.
    """

    ordered = list(cuis)
    if not ordered:
        raise ValueError("cuis must not be empty")
    if len(set(ordered)) != len(ordered):
        raise ValueError(f"cuis must be unique, got {ordered!r}")
    if not isinstance(answer, str) or not answer.strip():
        raise SelectionError("answer is empty or not a string")

    found = _from_json(answer, ordered)
    if not found:
        found = _from_mentions(answer, ordered)
    if not found:
        raise SelectionError(
            "answer did not mention any candidate CUI "
            f"(answer: {answer.strip()[:200]!r})"
        )

    seen: set[str] = set()
    ranking: list[str] = []
    for cui in found:
        if cui not in seen:
            seen.add(cui)
            ranking.append(cui)
    ranking.extend(cui for cui in ordered if cui not in seen)
    return ranking
