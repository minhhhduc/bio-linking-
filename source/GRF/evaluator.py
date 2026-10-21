"""C9: pipeline evaluation (R@1/R@5/accuracy) for the GRF linker.

The metric computations themselves live in :mod:`common.evaluator`; this module
only builds requests from a :class:`PipelineConfig` and runs the linker.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from common.evaluator import (  # noqa: F401 — re-exported
    Metrics,
    _metrics,
    metrics_from_responses,
)
from common.data import MentionSample
from common.schemas import Request, Response

from .config import PipelineConfig


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
    """R@1, R@5 and end-to-end accuracy for ``samples``."""
    sample_list = list(samples)
    return _metrics(sample_list, _link(linker, sample_list))


def evaluate_all(
    linker,
    samples: Iterable[MentionSample],
    *,
    progress_callback=None,
) -> tuple[Metrics, dict[str | None, Metrics]]:
    """Overall and per-label metrics from **one** pipeline run."""
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
