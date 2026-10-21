"""Candidate-retrieval snapshots shared by the retrieval and E2E notebooks.

One JSONL file stores one record per gold mention: the sample fields needed for
evaluation plus the retrieved candidates and the untruncated concept ranking.
The E2E stage re-ranks these candidates without loading the encoder or the
vector index again.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

from .data import MentionSample
from .rerank import Reranker, select_with_fallback
from .schemas import Candidate, Response

FORMAT = "grf-candidates-v1"
_REQUIRED_KEYS = (
    "format",
    "mention",
    "context",
    "gold_cui",
    "record_index",
    "entity_index",
    "cui",
    "candidates",
)


def _candidate_from_record(item: object, where: str) -> Candidate:
    if not isinstance(item, dict):
        raise ValueError(f"{where}: candidate must be an object")
    try:
        cui = item["cui"]
        score = item["score"]
        surface = item["surface"]
    except KeyError as exc:
        raise ValueError(f"{where}: missing key {exc.args[0]!r}") from exc
    if not isinstance(cui, str) or not cui.strip():
        raise ValueError(f"{where}: cui must be a non-empty string")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError(f"{where}: score must be a number")
    if not isinstance(surface, str) or not surface.strip():
        raise ValueError(f"{where}: surface must be a non-empty string")
    return Candidate(cui=cui, score=float(score), surface=surface)


def _optional_int(record: dict, key: str, where: str) -> int | None:
    value = record.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{where}: {key} must be an integer or null")
    return value


def save_candidates(
    path: str | Path,
    samples: Sequence[MentionSample],
    responses: Sequence[Response],
    *,
    progress_callback=None,
) -> None:
    """Write samples and their retrieval responses to one JSONL file."""
    if progress_callback is not None and not callable(progress_callback):
        raise ValueError("progress_callback must be callable or None")
    sample_list = list(samples)
    response_list = list(responses)
    if len(sample_list) != len(response_list):
        raise ValueError(
            f"got {len(sample_list)} samples but {len(response_list)} responses"
        )

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for index, (sample, response) in enumerate(
            zip(sample_list, response_list)
        ):
            if not isinstance(sample, MentionSample):
                raise ValueError(f"samples[{index}] must be a MentionSample")
            if not isinstance(response, Response):
                raise ValueError(f"responses[{index}] must be a Response")
            debug = response.debug if isinstance(response.debug, dict) else {}
            ranking = debug.get("concept_ranking")
            record = {
                "format": FORMAT,
                "mention": sample.mention,
                "context": sample.context,
                "gold_cui": sample.gold_cui,
                "label": sample.label,
                "start_token": sample.start_token,
                "end_token": sample.end_token,
                "record_index": sample.record_index,
                "entity_index": sample.entity_index,
                "cui": response.cui,
                "candidates": [asdict(candidate) for candidate in response.candidates],
                "concept_ranking": ranking if isinstance(ranking, list) else [],
                "score_kind": debug.get("score_kind"),
                "error": debug.get("error"),
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if progress_callback is not None:
                progress_callback(index + 1, len(sample_list))


def load_candidates(
    path: str | Path, *, progress_callback=None, total: int | None = None
) -> tuple[list[MentionSample], list[Response]]:
    """Read a file written by :func:`save_candidates`.

    ``progress_callback(done, total)`` is called after each record is parsed;
    pass ``total`` when the caller has already counted the records, otherwise
    it is ``None``.
    """
    if progress_callback is not None and not callable(progress_callback):
        raise ValueError("progress_callback must be callable or None")
    if total is not None and (
        isinstance(total, bool) or not isinstance(total, int) or total < 0
    ):
        raise ValueError("total must be a non-negative integer or None")
    source = Path(path)
    samples: list[MentionSample] = []
    responses: list[Response] = []
    with source.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            where = f"{source}:{line_number}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{where}: invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{where}: record must be an object")
            for key in _REQUIRED_KEYS:
                if key not in record:
                    raise ValueError(f"{where}: missing key {key!r}")
            if record["format"] != FORMAT:
                raise ValueError(
                    f"{where}: unsupported format {record['format']!r}; "
                    f"expected {FORMAT!r}"
                )
            for key in ("mention", "context", "gold_cui", "cui"):
                if not isinstance(record[key], str):
                    raise ValueError(f"{where}: {key} must be a string")

            raw_candidates = record["candidates"]
            if not isinstance(raw_candidates, list):
                raise ValueError(f"{where}: candidates must be a list")
            candidates = tuple(
                _candidate_from_record(item, f"{where}:candidates[{index}]")
                for index, item in enumerate(raw_candidates)
            )

            raw_ranking = record.get("concept_ranking") or []
            if not isinstance(raw_ranking, list):
                raise ValueError(f"{where}: concept_ranking must be a list")
            ranking = [
                asdict(_candidate_from_record(item, f"{where}:concept_ranking[{index}]"))
                for index, item in enumerate(raw_ranking)
            ]

            label = record.get("label")
            if label is not None and not isinstance(label, str):
                raise ValueError(f"{where}: label must be a string or null")
            for key in ("record_index", "entity_index"):
                value = record[key]
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ValueError(f"{where}: {key} must be an integer")
            sample = MentionSample(
                mention=record["mention"],
                context=record["context"],
                gold_cui=record["gold_cui"],
                label=label,
                start_token=_optional_int(record, "start_token", where),
                end_token=_optional_int(record, "end_token", where),
                record_index=record["record_index"],
                entity_index=record["entity_index"],
            )
            debug = {
                "concept_ranking": ranking,
                "score_kind": record.get("score_kind"),
                "error": record.get("error"),
            }
            response = Response(
                cui=record["cui"],
                candidates=candidates,
                debug=debug,
            )
            samples.append(sample)
            responses.append(response)
            if progress_callback is not None:
                progress_callback(len(samples), total)
    return samples, responses


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
