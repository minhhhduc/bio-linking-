"""GRF re-ranking of saved candidate snapshots.

Candidate file I/O lives in :mod:`common.candidates`; this module re-exports it
and adds the GRF re-ranker, which is the only GRF-specific part.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from common.candidates import (  # noqa: F401 — re-exported
    FORMAT,
    load_candidates,
    save_candidates,
)
from common.data import MentionSample
from common.schemas import Response

from .rerank import RerankError, Reranker, select_with_fallback


def rerank_candidates(
    samples: Sequence[MentionSample],
    responses: Sequence[Response],
    *,
    reranker: Reranker,
    concept_names: Mapping[str, str] | None = None,
    progress_callback=None,
) -> list[Response]:
    """Re-rank saved candidates; LLM failures fall back to Top-1.

    ``progress_callback(done, total)`` is called after each sample for the
    Colab progress bar.

    A re-ranker that offers ``rerank_batch`` is driven through it: every
    mention's prompt is decoded in one batched call instead of one call per
    mention, which is what the local client needs to use the GPU at all.  The
    per-mention fallback to Top-1 is unchanged either way.
    """
    if progress_callback is not None and not callable(progress_callback):
        raise ValueError("progress_callback must be callable or None")
    sample_list = list(samples)
    response_list = list(responses)
    if len(sample_list) != len(response_list):
        raise ValueError(
            f"got {len(sample_list)} samples but {len(response_list)} responses"
        )
    names = concept_names if concept_names is not None else {}

    for index, (sample, response) in enumerate(zip(sample_list, response_list)):
        if not isinstance(sample, MentionSample):
            raise ValueError(f"samples[{index}] must be a MentionSample")
        if not isinstance(response, Response):
            raise ValueError(f"responses[{index}] must be a Response")

    batch = getattr(reranker, "rerank_batch", None)
    batched: list[str | Exception] | None = None
    if batch is not None:
        todo = [
            index
            for index, response in enumerate(response_list)
            if response.candidates
        ]
        if todo:
            try:
                outcomes = list(
                    batch(
                        [
                            (
                                sample_list[index].mention,
                                sample_list[index].context,
                                response_list[index].candidates,
                                names,
                            )
                            for index in todo
                        ]
                    )
                )
            except Exception:
                # One unusable prompt (or a batch-level failure) must not cost
                # the whole run: fall back to the per-mention path below,
                # which isolates each failure.
                batched = None
            else:
                if len(outcomes) != len(todo):
                    batched = None
                else:
                    batched = [None] * len(response_list)
                    for index, outcome in zip(todo, outcomes):
                        batched[index] = outcome

    reranked: list[Response] = []
    for done, (sample, response) in enumerate(
        zip(sample_list, response_list), start=1
    ):
        index = done - 1
        debug = dict(response.debug) if isinstance(response.debug, dict) else {}
        if not response.candidates:
            debug["rerank"] = {"error": "no_candidates"}
            reranked.append(
                Response(cui=response.cui, candidates=response.candidates, debug=debug)
            )
        elif batched is not None:
            outcome = batched[index]
            valid = {candidate.cui for candidate in response.candidates}
            if isinstance(outcome, str) and outcome in valid:
                cui, error = outcome, None
            else:
                reason = (
                    str(outcome)
                    if isinstance(outcome, RerankError)
                    else f"re-ranker returned {outcome!r}, which is not a candidate"
                )
                cui, error = response.candidates[0].cui, reason
            debug["rerank"] = {"error": error}
            reranked.append(
                Response(cui=cui, candidates=response.candidates, debug=debug)
            )
        else:
            cui, error = select_with_fallback(
                reranker,
                sample.mention,
                sample.context,
                response.candidates,
                names,
            )
            debug["rerank"] = {"error": error}
            reranked.append(
                Response(cui=cui, candidates=response.candidates, debug=debug)
            )
        if progress_callback is not None:
            progress_callback(done, len(sample_list))
    return reranked
