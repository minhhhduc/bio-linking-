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


def interpolate_matrix(
    q0_matrix: np.ndarray,
    components: Sequence[np.ndarray],
    weights: Sequence[Sequence[float]],
) -> np.ndarray:
    """``Q = W · [Q_0; Q_F1 … Q_Fn]`` for a whole batch in one contraction.

    ``q0_matrix`` is ``(b, d)``; ``components[i]`` is the ``(n_i, d)`` block of
    feedback vectors of row ``i`` (``n_i`` can differ per row); ``weights[i]``
    holds the ``n_i + 1`` weights of that row, ``Q_0`` first.  Rows are
    L2-normalised, like :func:`interpolate_vectors`.

    Ragged feedback counts are padded with zero vectors *and* zero weights, so
    they contribute nothing to the sum.
    """
    base = np.asarray(q0_matrix, dtype=np.float64)
    if base.ndim != 2:
        raise ValueError(f"q0_matrix must be 2-D, got shape {base.shape!r}")
    if not np.all(np.isfinite(base)):
        raise ValueError("q0_matrix contains non-finite values")

    vector_blocks = list(components)
    weight_rows = list(weights)
    rows, dim = base.shape
    if len(vector_blocks) != rows or len(weight_rows) != rows:
        raise ValueError(
            f"expected {rows} component/weight rows, got "
            f"{len(vector_blocks)} and {len(weight_rows)}"
        )

    width = 1 + max((len(block) for block in vector_blocks), default=0)
    stacked = np.zeros((rows, width, dim), dtype=np.float64)
    stacked[:, 0] = base
    scale = np.zeros((rows, width), dtype=np.float64)
    for row, (block, row_weights) in enumerate(zip(vector_blocks, weight_rows)):
        arrays = np.asarray(block, dtype=np.float64)
        if arrays.size == 0:
            arrays = np.zeros((0, dim), dtype=np.float64)
        if arrays.shape != (len(row_weights) - 1, dim):
            raise ValueError(
                f"row {row}: {arrays.shape[0]} feedback vectors for "
                f"{len(row_weights) - 1} weights"
            )
        if not np.all(np.isfinite(arrays)):
            raise ValueError("feedback vectors contain non-finite values")
        count = arrays.shape[0]
        stacked[row, 1 : 1 + count] = arrays
        weights_array = np.asarray(row_weights, dtype=np.float64)
        if weights_array.ndim != 1 or weights_array.shape[0] != count + 1:
            raise ValueError(
                f"row {row}: {count + 1} weights expected, got "
                f"{weights_array.shape!r}"
            )
        if not np.all(np.isfinite(weights_array)):
            raise ValueError("weights contain non-finite values")
        scale[row, : count + 1] = weights_array

    norms_check = scale.sum(axis=1)
    if np.any(np.abs(norms_check - 1.0) > _WEIGHT_TOLERANCE):
        raise ValueError("every weight row must sum to 1")

    fused = np.einsum("rx,rxd->rd", scale, stacked)
    norms = np.linalg.norm(fused, axis=1, keepdims=True)
    if not np.all(np.isfinite(norms)) or np.any(norms <= 0.0):
        raise ValueError("fused vector has zero norm")
    return (fused / norms).astype(np.float32)


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


def rrf_score_matrix(
    cui_ids: np.ndarray,
    ranks: np.ndarray,
    weights,
    row_ids: np.ndarray,
    n_rows: int,
    n_concepts: int,
) -> np.ndarray:
    """Dense RRF scores for a batch of mentions, one scatter-add.

    ``cui_ids`` / ``ranks`` are ``(q, k)``: concept column and 1-based rank of
    the k-th entry of each query's concept rank list (``cui_ids < 0`` marks an
    unused slot).  ``weights`` holds the ``q`` query weights and ``row_ids``
    the mention each query belongs to.  Returns ``(n_rows, n_concepts)`` with

    ``S[r, c] = Σ_x w_x / r_x(c)``  and 0 where ``c`` is absent from every list
    of row ``r`` (the documented ``1/r_x(c) = 0`` convention).
    """
    ids = np.asarray(cui_ids, dtype=np.int64)
    order = np.asarray(ranks, dtype=np.float64)
    if ids.ndim != 2 or ids.shape != order.shape:
        raise ValueError("cui_ids and ranks must be matching 2-D arrays")
    queries = ids.shape[0]
    owner = np.asarray(row_ids, dtype=np.int64)
    scale = np.asarray(weights, dtype=np.float64)
    if owner.shape != (queries,) or scale.shape != (queries,):
        raise ValueError(
            f"row_ids and weights must have shape ({queries},), got "
            f"{owner.shape!r} and {scale.shape!r}"
        )
    if not np.all(np.isfinite(scale)) or np.any(scale < 0.0):
        raise ValueError("weights must be finite and non-negative")
    if queries and (owner.min() < 0 or owner.max() >= n_rows):
        raise ValueError("row_ids out of range")
    present = ids >= 0
    if np.any(ids[present] >= n_concepts):
        raise ValueError("cui_ids out of range for n_concepts")
    if not np.all(np.isfinite(order)) or np.any(order[present] < 1.0):
        raise ValueError("ranks must be finite and >= 1")

    contribution = np.where(
        present, scale[:, None] / np.where(present, order, 1.0), 0.0
    )
    flat = owner[:, None] * n_concepts + np.where(present, ids, 0)
    return np.bincount(
        flat.ravel(),
        weights=contribution.ravel(),
        minlength=n_rows * n_concepts,
    ).reshape(n_rows, n_concepts)
