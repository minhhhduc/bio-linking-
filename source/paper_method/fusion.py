"""C6: fusion of the original query with GRF feedback components."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

_WEIGHT_TOLERANCE = 1e-6


def component_weights(
    n_feedback_components: int,
) -> tuple[float, tuple[float, ...]]:
    """Return ``(w0, (w_f1, ...))`` with every weight equal to ``1/(n+1)``."""
    if isinstance(n_feedback_components, bool) or not isinstance(
        n_feedback_components, (int, np.integer)
    ):
        raise TypeError("n_feedback_components must be an integer")
    if n_feedback_components < 0:
        raise ValueError("n_feedback_components must be >= 0")

    weight = 1.0 / (int(n_feedback_components) + 1)
    return weight, tuple(weight for _ in range(int(n_feedback_components)))


def _checked_weights(weights, expected: int) -> np.ndarray:
    if isinstance(weights, (str, bytes)):
        raise ValueError("weights must be numeric")
    try:
        items = list(weights)
    except TypeError as exc:
        raise ValueError("weights must be a sequence") from exc
    if any(isinstance(item, (bool, np.bool_)) for item in items):
        raise ValueError("weights must not contain booleans")
    try:
        array = np.asarray(items, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("weights must be numeric") from exc
    if array.ndim != 1:
        raise ValueError("weights must be a 1-D sequence")
    if array.shape[0] != expected:
        raise ValueError(
            f"expected {expected} weights, got {array.shape[0]}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("weights contain non-finite values")
    if np.any(array < 0.0):
        raise ValueError("weights must be non-negative")
    if abs(float(array.sum()) - 1.0) > _WEIGHT_TOLERANCE:
        raise ValueError(
            f"weights must sum to 1 (got {float(array.sum())!r})"
        )
    return array


def interpolate_vectors(
    q0,
    feedback_vectors: Sequence[np.ndarray],
    weights,
) -> np.ndarray:
    """Weighted sum of ``q0`` and feedback vectors, L2-normalised."""
    base = np.asarray(q0, dtype=np.float64)
    if base.ndim != 1:
        raise ValueError(f"q0 must be 1-D, got shape {base.shape!r}")
    if not np.all(np.isfinite(base)):
        raise ValueError("q0 contains non-finite values")

    try:
        feedback = list(feedback_vectors)
    except TypeError as exc:
        raise ValueError("feedback_vectors must be a sequence") from exc

    arrays: list[np.ndarray] = []
    for vector in feedback:
        array = np.asarray(vector, dtype=np.float64)
        if array.ndim != 1 or array.shape != base.shape:
            raise ValueError(
                "every feedback vector must have the same 1-D shape as q0"
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("feedback vectors contain non-finite values")
        arrays.append(array)

    checked = _checked_weights(weights, 1 + len(arrays))
    fused = checked[0] * base
    for weight, vector in zip(checked[1:], arrays):
        fused = fused + weight * vector

    norm = float(np.linalg.norm(fused))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("fused vector has zero norm")
    return (fused / norm).astype(np.float32)


def rrf_scores(
    rank_lists: Sequence[Mapping[str, int]],
    weights,
) -> dict[str, float]:
    """Weighted reciprocal rank fusion over ``{cui: 1-based rank}`` lists."""
    try:
        lists = list(rank_lists)
    except TypeError as exc:
        raise ValueError("rank_lists must be a sequence") from exc

    checked = _checked_weights(weights, len(lists))
    scores: dict[str, float] = {}

    for ranking, weight in zip(lists, checked):
        if not isinstance(ranking, Mapping):
            raise ValueError("each rank list must be a {cui: rank} mapping")
        for cui, rank in ranking.items():
            if not isinstance(cui, str) or not cui.strip():
                raise ValueError("rank list keys must be non-empty CUI strings")
            if (
                isinstance(rank, bool)
                or not isinstance(rank, (int, np.integer))
                or rank < 1
            ):
                raise ValueError("ranks must be integers >= 1")
            scores[cui] = scores.get(cui, 0.0) + float(weight) / float(rank)

    return scores
