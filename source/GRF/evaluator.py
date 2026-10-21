"""C9: candidate-retrieval and end-to-end normalization metrics.

Architecture §10 asks for R@1/R@5 on candidate retrieval and accuracy after
re-ranking.  The only gold data available here is the Vietnamese BEL JSONL;
its samples carry one gold CUI per annotated mention.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .config import PipelineConfig
from .data import MentionSample
from .schemas import Request, Response


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


def make_requests(
    samples: Iterable[MentionSample], config: PipelineConfig
) -> list[Request]:
    """Build one pipeline request per annotated mention."""
    if not isinstance(config, PipelineConfig):
        raise TypeError("config must be a PipelineConfig")
    return [
        Request(
            mention=sample.mention,
            context=sample.context,
            feedback_types=tuple(config.feedback_types),
            n_syn=config.n_syn,
        )
        for sample in samples
    ]


def _retrieval_cuis(response: Response) -> list[str]:
    """Concept ranking before the re-ranker's ``top_k_concept`` truncation.

    ``Response.candidates`` is deliberately truncated for re-ranking; using it
    for R@5 would cap recall at ``top_k_concept``.  The pipeline therefore also
    records the full retrieval ranking in ``debug["concept_ranking"]``.
    Custom linkers that do not provide it fall back to ``candidates``.
    """
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
    """Overall and per-label metrics for already-computed responses.

    Used by the retrieval notebook (metrics after one ``link_batch``) and by
    the end-to-end notebook (metrics after re-ranking a saved candidate file).
    """
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


def _link(
    linker,
    samples: Sequence[MentionSample],
    progress_callback=None,
) -> Sequence[Response]:
    """Run the pipeline once and enforce a one-response-per-sample contract."""
    requests = make_requests(samples, linker.config)
    if progress_callback is None:
        responses = linker.link_batch(requests)
    else:
        responses = linker.link_batch(requests, progress_callback=progress_callback)
    if len(responses) != len(samples):
        raise ValueError(
            "linker.link_batch returned "
            f"{len(responses)} responses for {len(samples)} requests"
        )
    return responses


def evaluate(linker, samples: Iterable[MentionSample]) -> Metrics:
    """R@1, R@5 and end-to-end accuracy for ``samples``.

    Requests use ``linker.config.feedback_types`` and ``linker.config.n_syn``,
    so the GRF configuration is the one the linker was built with.  Unexpected
    errors from the pipeline propagate; only the pipeline's documented
    degradation paths (LLM failures, empty candidates) are tolerated.
    """
    sample_list = list(samples)
    return _metrics(sample_list, _link(linker, sample_list))


def evaluate_all(
    linker,
    samples: Iterable[MentionSample],
    *,
    progress_callback=None,
) -> tuple[Metrics, dict[str | None, Metrics]]:
    """Overall and per-label metrics from **one** pipeline run.

    Evaluating overall and by label separately would call the LLM twice; since
    failed feedback calls are deliberately not cached, the two reports could
    then disagree (and every request would pay twice).

    ``progress_callback(done, total)`` is forwarded to ``link_batch`` for
    per-sample progress reporting in the Colab notebooks.
    """
    if progress_callback is not None and not callable(progress_callback):
        raise ValueError("progress_callback must be callable or None")
    sample_list = list(samples)
    responses = _link(linker, sample_list, progress_callback)
    return metrics_from_responses(sample_list, responses)


def evaluate_by_label(
    linker, samples: Iterable[MentionSample]
) -> dict[str | None, Metrics]:
    """Group metrics by the gold entity label, preserving first-seen order."""
    return evaluate_all(linker, samples)[1]
