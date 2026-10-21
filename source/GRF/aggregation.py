"""C7: aggregate surface-form hits into concept-level candidates."""

from __future__ import annotations

import math
from numbers import Real

import numpy as np

from .schemas import Candidate


def _validated_hits(hits) -> list:
    try:
        items = list(hits)
    except TypeError as exc:
        raise ValueError("hits must be a sequence of RetrievalHit") from exc

    for hit in items:
        if not all(
            hasattr(hit, field) for field in ("cui", "surface", "score", "rank")
        ):
            raise ValueError(
                "each hit must provide cui, surface, score and rank"
            )
        cui = hit.cui
        surface = hit.surface
        if not isinstance(cui, str) or not cui.strip():
            raise ValueError("hit.cui must be a non-empty string")
        if not isinstance(surface, str) or not surface.strip():
            raise ValueError("hit.surface must be a non-empty string")

        score = hit.score
        if isinstance(score, bool) or not isinstance(
            score, (Real, np.integer, np.floating)
        ):
            raise ValueError("hit.score must be a real number")
        if not math.isfinite(float(score)):
            raise ValueError("hit.score must be finite")

        rank = hit.rank
        if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)) or rank < 1:
            raise ValueError("hit.rank must be an integer >= 1")

    return items


def _concept_scores(hits) -> dict[str, tuple[float, str]]:
    """Map CUI -> (max score, representative surface) deterministically."""
    best: dict[str, tuple[float, str]] = {}
    for hit in _validated_hits(hits):
        cui = hit.cui
        score = float(hit.score)
        surface = hit.surface
        current = best.get(cui)
        if current is None or score > current[0] or (
            score == current[0] and surface < current[1]
        ):
            best[cui] = (score, surface)
    return best


def aggregate_hits(hits, top_k_concept: int | None = None) -> list[Candidate]:
    """``Score(c) = max S(a)`` over retrieved aliases, sorted score/cui."""
    if top_k_concept is not None and (
        isinstance(top_k_concept, bool)
        or not isinstance(top_k_concept, (int, np.integer))
        or top_k_concept <= 0
    ):
        raise ValueError("top_k_concept must be None or a positive integer")

    best = _concept_scores(hits)
    ordered = sorted(best.items(), key=lambda item: (-item[1][0], item[0]))
    if top_k_concept is not None:
        ordered = ordered[:top_k_concept]

    return [
        Candidate(cui=cui, score=score, surface=surface)
        for cui, (score, surface) in ordered
    ]


def concept_rank_list(hits) -> dict[str, int]:
    """Return ``{cui: 1-based rank}`` using the aggregation ordering."""
    candidates = aggregate_hits(hits)
    return {candidate.cui: rank for rank, candidate in enumerate(candidates, start=1)}
