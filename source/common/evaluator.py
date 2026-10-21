"""C9 metrics: R@1/R@5 on retrieval plus end-to-end accuracy.

These functions operate on already-computed :class:`Response` objects, so the
same code serves the GRF pipeline and the dense baseline.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .data import MentionSample
from .schemas import Response


@dataclass(frozen=True)
class Metrics:
    """Metrics over one sample group (``n == 0`` means no samples)."""

    n: int
    recall_at_1: float
    recall_at_5: float
    accuracy: float

    def __post_init__(self) -> None:
        if isinstance(self.n, bool) or not isinstance(self.n, int) or self.n < 0:
            raise ValueError("n must be a non-negative integer")
        for name in ("recall_at_1", "recall_at_5", "accuracy"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a number")
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value!r}")
        if self.n == 0 and (self.recall_at_1 or self.recall_at_5 or self.accuracy):
            raise ValueError("empty metrics must be zero")


def _retrieval_cuis(response: Response) -> list[str]:
    """Concept ranking before the re-ranker's ``top_k_concept`` truncation."""
    def valid_cuis(ranking) -> list[str] | None:
        if not isinstance(ranking, list) or not ranking:
            return None
        cuis: list[str] = []
        for item in ranking:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("cui"), str)
                or not item["cui"]
            ):
                return None
            cuis.append(item["cui"])
        return cuis

    debug = response.debug if isinstance(response.debug, dict) else {}
    cuis = valid_cuis(debug.get("concept_ranking"))
    if cuis is None:
        retrieval = debug.get("retrieval")
        if isinstance(retrieval, dict):
            cuis = valid_cuis(retrieval.get("concept_ranking"))
    if cuis is not None:
        return cuis
    return [candidate.cui for candidate in response.candidates]


def _metrics(samples: Sequence[MentionSample], responses: Sequence[Response]) -> Metrics:
    n = len(samples)
    if n == 0:
        return Metrics(n=0, recall_at_1=0.0, recall_at_5=0.0, accuracy=0.0)

    recall_1 = 0
    recall_5 = 0
    correct = 0
    for sample, response in zip(samples, responses):
        cuis = _retrieval_cuis(response)
        if sample.gold_cui in cuis[:1]:
            recall_1 += 1
        if sample.gold_cui in cuis[:5]:
            recall_5 += 1
        if response.cui == sample.gold_cui:
            correct += 1

    return Metrics(
        n=n,
        recall_at_1=recall_1 / n,
        recall_at_5=recall_5 / n,
        accuracy=correct / n,
    )


def metrics_from_responses(
    samples: Iterable[MentionSample],
    responses: Iterable[Response],
) -> tuple[Metrics, dict[str | None, Metrics]]:
    """Overall and per-label metrics for already-computed responses."""
    sample_list = list(samples)
    if isinstance(responses, (str, bytes)):
        raise ValueError("responses must be a sequence of Response objects")
    response_list = list(responses)
    if len(sample_list) != len(response_list):
        raise ValueError(
            f"got {len(sample_list)} samples but {len(response_list)} responses"
        )
    for index, response in enumerate(response_list):
        if not isinstance(response, Response):
            raise ValueError(f"responses[{index}] must be a Response")

    groups: dict[str | None, tuple[list[MentionSample], list[Response]]] = {}
    for sample, response in zip(sample_list, response_list):
        group_samples, group_responses = groups.setdefault(sample.label, ([], []))
        group_samples.append(sample)
        group_responses.append(response)

    overall = _metrics(sample_list, response_list)
    by_label = {
        label: _metrics(group_samples, group_responses)
        for label, (group_samples, group_responses) in groups.items()
    }
    return overall, by_label
