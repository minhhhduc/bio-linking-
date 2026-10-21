"""Loading and validation of gold BEL JSONL records.

The gold file format is one JSON object per line::

    {"entities": [{"cui": ..., "label": ..., "vi_term": ...,
                   "vi_start_token_idx": ..., "vi_end_token_idx": ...}],
     "vi_sentence_str": "..."}

``vi_term`` is the mention, while token indices are optional metadata used
only to sanity-check the mention against the whitespace tokenisation of the
context (architecture §4.3 input).  Structural problems are collected as
:class:`DataIssue` entries; with ``strict=True`` they raise
:class:`DataValidationError`.  Span problems are never fatal and never change
or drop the sample.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class DataValidationError(ValueError):
    """Raised for malformed JSON or, in strict mode, structural record errors."""


@dataclass(frozen=True)
class MentionSample:
    """One entity mention extracted from a gold record."""

    mention: str
    context: str
    gold_cui: str
    label: str | None
    start_token: int | None
    end_token: int | None
    record_index: int
    entity_index: int


@dataclass(frozen=True)
class DataIssue:
    """A non-fatal (or, when raised, reported) validation finding."""

    record_index: int
    entity_index: int | None
    code: str
    detail: str


@dataclass(frozen=True)
class DatasetLoadResult:
    """Samples plus every issue found while loading."""

    samples: tuple[MentionSample, ...]
    issues: tuple[DataIssue, ...]


def _normalize(text: str) -> str:
    """NFC-normalise and collapse whitespace so Vietnamese strings compare."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def _is_int(value: Any) -> bool:
    """``bool`` is an ``int`` subclass but is never a valid token index."""
    return isinstance(value, int) and not isinstance(value, bool)


def _structural_error(
    *,
    strict: bool,
    issues: list[DataIssue],
    record_index: int,
    entity_index: int | None,
    code: str,
    detail: str,
    line_number: int,
) -> None:
    if strict:
        location = f"line {line_number}, record {record_index}"
        if entity_index is not None:
            location += f", entity {entity_index}"
        raise DataValidationError(f"{location}: {detail}")
    issues.append(DataIssue(record_index, entity_index, code, detail))


def _read_span(
    entity: dict[str, Any],
    *,
    tokens: list[str],
    mention: str,
    record_index: int,
    entity_index: int,
    issues: list[DataIssue],
) -> tuple[int | None, int | None]:
    """Validate optional token indices.

    Returns the span when both indices are usable, otherwise ``(None, None)``.
    Bad spans add a ``bad_span`` issue; a well-formed span whose text disagrees
    with ``vi_term`` adds ``span_mismatch`` but the indices are still returned.
    """
    start = entity.get("vi_start_token_idx")
    end = entity.get("vi_end_token_idx")
    start_present = start is not None
    end_present = end is not None

    if not start_present and not end_present:
        return None, None

    if start_present != end_present:
        issues.append(DataIssue(
            record_index, entity_index, "bad_span",
            "only one of vi_start_token_idx / vi_end_token_idx is present",
        ))
        return None, None

    if not _is_int(start) or not _is_int(end):
        issues.append(DataIssue(
            record_index, entity_index, "bad_span",
            f"token indices must be integers, got {start!r} and {end!r}",
        ))
        return None, None

    assert isinstance(start, int) and isinstance(end, int)
    if start < 0 or end < 0:
        issues.append(DataIssue(
            record_index, entity_index, "bad_span",
            f"token indices must be non-negative, got ({start}, {end})",
        ))
        return None, None
    if end < start:
        issues.append(DataIssue(
            record_index, entity_index, "bad_span",
            f"vi_end_token_idx ({end}) is before vi_start_token_idx ({start})",
        ))
        return None, None
    if start >= len(tokens) or end >= len(tokens):
        issues.append(DataIssue(
            record_index, entity_index, "bad_span",
            f"token indices ({start}, {end}) outside context with "
            f"{len(tokens)} tokens",
        ))
        return None, None

    observed = _normalize(" ".join(tokens[start:end + 1]))
    expected = _normalize(mention)
    if observed != expected:
        issues.append(DataIssue(
            record_index, entity_index, "span_mismatch",
            f"context span {observed!r} != vi_term {expected!r}",
        ))
    return start, end


def load_bel_jsonl(
    path: str | Path, *, strict: bool = False
) -> DatasetLoadResult:
    """Load gold BEL samples from a JSONL file.

    ``record_index`` is the 0-based index of the record among non-blank lines;
    ``entity_index`` is the 0-based position inside its ``entities`` list.
    Invalid JSON always raises :class:`DataValidationError` with the line
    number.  With ``strict=True``, structural errors (non-object record,
    unusable ``entities``, missing/empty ``vi_sentence_str``, missing
    ``cui``/``vi_term``, bad label type) also raise.  Otherwise they become
    ``bad_record`` / ``bad_entity`` issues and the affected unit is skipped.
    Span issues are always non-fatal and never drop a structurally valid
    sample.
    """
    path = Path(path)
    samples: list[MentionSample] = []
    issues: list[DataIssue] = []
    record_index = -1

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record_index += 1
            ridx = record_index

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DataValidationError(
                    f"line {line_number}: invalid JSON: {exc.msg}"
                ) from exc

            if not isinstance(record, dict):
                _structural_error(
                    strict=strict, issues=issues,
                    record_index=ridx, entity_index=None,
                    code="bad_record",
                    detail="record must be a JSON object",
                    line_number=line_number,
                )
                continue

            context = record.get("vi_sentence_str")
            if not isinstance(context, str) or not context.strip():
                _structural_error(
                    strict=strict, issues=issues,
                    record_index=ridx, entity_index=None,
                    code="bad_record",
                    detail="missing or empty 'vi_sentence_str'",
                    line_number=line_number,
                )
                continue

            entities = record.get("entities")
            if not isinstance(entities, list):
                _structural_error(
                    strict=strict, issues=issues,
                    record_index=ridx, entity_index=None,
                    code="bad_record",
                    detail="'entities' must be a list",
                    line_number=line_number,
                )
                continue

            tokens = context.split()
            for entity_index, entity in enumerate(entities):
                if not isinstance(entity, dict):
                    _structural_error(
                        strict=strict, issues=issues,
                        record_index=ridx, entity_index=entity_index,
                        code="bad_entity",
                        detail="entity must be a JSON object",
                        line_number=line_number,
                    )
                    continue

                cui = entity.get("cui")
                if not isinstance(cui, str) or not cui.strip():
                    _structural_error(
                        strict=strict, issues=issues,
                        record_index=ridx, entity_index=entity_index,
                        code="bad_entity",
                        detail="missing or empty 'cui'",
                        line_number=line_number,
                    )
                    continue

                mention = entity.get("vi_term")
                if not isinstance(mention, str) or not mention.strip():
                    _structural_error(
                        strict=strict, issues=issues,
                        record_index=ridx, entity_index=entity_index,
                        code="bad_entity",
                        detail="missing or empty 'vi_term'",
                        line_number=line_number,
                    )
                    continue

                label = entity.get("label")
                if label is not None and not isinstance(label, str):
                    _structural_error(
                        strict=strict, issues=issues,
                        record_index=ridx, entity_index=entity_index,
                        code="bad_entity",
                        detail=f"'label' must be a string or null, "
                               f"got {label!r}",
                        line_number=line_number,
                    )
                    continue

                start, end = _read_span(
                    entity,
                    tokens=tokens,
                    mention=mention,
                    record_index=ridx,
                    entity_index=entity_index,
                    issues=issues,
                )
                samples.append(MentionSample(
                    mention=_normalize(mention),
                    context=context,
                    gold_cui=unicodedata.normalize("NFC", cui.strip()),
                    label=label,
                    start_token=start,
                    end_token=end,
                    record_index=ridx,
                    entity_index=entity_index,
                ))

    return DatasetLoadResult(samples=tuple(samples), issues=tuple(issues))
