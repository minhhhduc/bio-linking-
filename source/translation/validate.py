"""Kiểm chứng tag và span của bản dịch."""

from __future__ import annotations

from dataclasses import dataclass

from .parse import (
    ParsedSentence,
    TokenSpanError,
    char_span_to_token_span,
    tokenize_en,
)
from .tagging import Entity

_TRIM_CHARS = " \t\r\n.,;:!?()[]{}\"'"
_BOUNDARY_PUNCTUATION = _TRIM_CHARS + "-–—"


@dataclass(frozen=True)
class ValidatedEntity:
    en_term: str
    en_start_token_idx: int
    en_end_token_idx: int


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    reasons: tuple[str, ...]
    entities: tuple[ValidatedEntity, ...]

    @property
    def en_terms(self) -> tuple[str, ...]:
        return tuple(entity.en_term for entity in self.entities)


def validate_translation(
    entities: list[Entity] | tuple[Entity, ...],
    parsed: ParsedSentence,
) -> ValidationResult:
    """Validate tag/span; trả lý do cụ thể nếu fail."""

    reasons: list[str] = []
    expected_ids = list(range(len(entities)))
    tags_by_start = sorted(parsed.tags, key=lambda tag: tag.char_start)
    actual_ids = [tag.tag_id for tag in tags_by_start]

    duplicates = sorted({tag_id for tag_id in actual_ids if actual_ids.count(tag_id) > 1})
    if duplicates:
        reasons.append(f"duplicate_tag: duplicated tag id(s) {duplicates}")

    expected_set = set(expected_ids)
    actual_set = set(actual_ids)
    missing = sorted(expected_set - actual_set)
    extra = sorted(actual_set - expected_set)
    if missing:
        reasons.append(f"missing_tag: missing tag id(s) {missing}")
    if extra:
        reasons.append(f"extra_tag: unexpected tag id(s) {extra}")
    if not missing and not extra and actual_ids != expected_ids:
        reasons.append(
            f"tag_order: expected order {expected_ids}, got {actual_ids}"
        )

    by_id = {tag.tag_id: tag for tag in parsed.tags}
    validated: list[ValidatedEntity] = []
    tokenized = tokenize_en(parsed.clean)

    for entity_index, entity in enumerate(entities):
        tag = by_id.get(entity_index)
        if tag is None:
            continue

        en_term = tag.text
        if not en_term or en_term != en_term.strip():
            reasons.append(
                f"empty_or_padded_span: e{entity_index} has empty or padded text"
            )
            continue

        if en_term[0] in _BOUNDARY_PUNCTUATION or en_term[-1] in _BOUNDARY_PUNCTUATION:
            reasons.append(
                f"punctuated_span: e{entity_index} term {en_term!r} "
                "has punctuation at its boundary"
            )
            continue

        if tag.char_end <= tag.char_start:
            reasons.append(f"empty_or_padded_span: e{entity_index} has empty char span")
            continue

        try:
            start_token, end_token = char_span_to_token_span(
                tokenized, tag.char_start, tag.char_end
            )
        except TokenSpanError as exc:
            reasons.append(f"{exc.code}: e{entity_index}: {exc.detail}")
            continue

        observed = parsed.clean[tag.char_start:tag.char_end]
        if observed != en_term:
            reasons.append(
                f"term_span_mismatch: e{entity_index}: "
                f"{observed!r} != {en_term!r}"
            )
            continue

        validated.append(
            ValidatedEntity(
                en_term=en_term,
                en_start_token_idx=start_token,
                en_end_token_idx=end_token,
            )
        )

    return ValidationResult(
        ok=not reasons,
        reasons=tuple(reasons),
        entities=tuple(validated),
    )
