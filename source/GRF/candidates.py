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

    ``progress_callback(done, total)`` is called after each sample, or after
    each 32-item chunk of the batched path, for the Colab progress bar.

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
            # ``rerank_batch`` returns only after the whole scenario is decoded,
            # so call it in chunks and report each one: the bar moves while the
            # LLM runs. 32 is a progress granularity, not a throughput knob —
            # the client still decodes its own batch size per generate call.
            chunk_size = 32
            try:
                outcomes: list[str | Exception] = []
                for start in range(0, len(todo), chunk_size):
                    chunk = todo[start : start + chunk_size]
                    outcomes.extend(
                        batch(
                            [
                                (
                                    sample_list[index].mention,
                                    sample_list[index].context,
                                    response_list[index].candidates,
                                    names,
                                )
                                for index in chunk
                            ]
                        )
                    )
                    if progress_callback is not None:
                        progress_callback(len(outcomes), len(sample_list))
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

    def placed(response: Response, chosen: str, debug: dict) -> Response:
        """The re-ranked list: the choice first, the rest in retrieval order."""
        if chosen not in {candidate.cui for candidate in response.candidates}:
            return Response(cui=chosen, candidates=response.candidates, debug=debug)
        first = tuple(c for c in response.candidates if c.cui == chosen)
        rest = tuple(c for c in response.candidates if c.cui != chosen)
        return Response(cui=chosen, candidates=first + rest, debug=debug)

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
            reranked.append(placed(response, cui, debug))
        else:
            cui, error = select_with_fallback(
                reranker,
                sample.mention,
                sample.context,
                response.candidates,
                names,
            )
            debug["rerank"] = {"error": error}
            reranked.append(placed(response, cui, debug))
        if progress_callback is not None and batched is None:
            progress_callback(done, len(sample_list))
    if (
        progress_callback is not None
        and batched is not None
        and len(todo) != len(sample_list)
    ):
        # Mentions without candidates never entered the batched call.
        progress_callback(len(sample_list), len(sample_list))
    return reranked
