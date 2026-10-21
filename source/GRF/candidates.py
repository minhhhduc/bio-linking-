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

from .rerank import Reranker, select_with_fallback


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
    """
    if progress_callback is not None and not callable(progress_callback):
        raise ValueError("progress_callback must be callable or None")
    sample_list = list(samples)
    response_list = list(responses)
    if len(sample_list) != len(response_list):
        raise ValueError(
            f"got {len(sample_list)} samples but {len(response_list)} responses"
        )

    reranked: list[Response] = []
    for done, (sample, response) in enumerate(
        zip(sample_list, response_list), start=1
    ):
        if not isinstance(sample, MentionSample):
            raise ValueError(f"samples[{done - 1}] must be a MentionSample")
        if not isinstance(response, Response):
            raise ValueError(f"responses[{done - 1}] must be a Response")
        debug = dict(response.debug) if isinstance(response.debug, dict) else {}
        if not response.candidates:
            debug["rerank"] = {"error": "no_candidates"}
            reranked.append(
                Response(cui=response.cui, candidates=response.candidates, debug=debug)
            )
        else:
            cui, error = select_with_fallback(
                reranker,
                sample.mention,
                sample.context,
                response.candidates,
                concept_names,
            )
            debug["rerank"] = {"error": error}
            reranked.append(
                Response(cui=cui, candidates=response.candidates, debug=debug)
            )
        if progress_callback is not None:
            progress_callback(done, len(sample_list))
    return reranked
