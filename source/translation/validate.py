"""Kiểm chứng tag, span và CUI của bản dịch."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .parse import (
    ParsedSentence,
    TokenSpanError,
    char_span_to_token_span,
    tokenize_en,
)
from .tagging import Entity

_TRIM_CHARS = " \t\r\n.,;:!?()[]{}\"'"
_BOUNDARY_PUNCTUATION = _TRIM_CHARS + "-–—"


class SapBertUnavailable(RuntimeError):
    """Không có sentence-transformers hoặc thiếu model SapBERT local."""


def normalize_term(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text).casefold()
    normalized = " ".join(normalized.split())
    return normalized.strip(_TRIM_CHARS)


def _singularize(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("ses"):
        # viruses -> virus, nhưng diseases -> disease, cases -> case.
        return word[:-2] if word[:-2].endswith("s") else word[:-1]
    if len(word) > 4 and word.endswith(("xes", "zes", "ches", "shes")):
        return word[:-2]
    if (
        len(word) > 3
        and word.endswith("s")
        and not word.endswith(("ss", "us", "is"))
    ):
        return word[:-1]
    return word


def canonical_term(text: str) -> str:
    """Chuẩn hoá nhẹ để chấp nhận biến thể số ít/số nhiều, hyphen, possessive."""

    normalized = normalize_term(text)
    # possessive: Alzheimer's / Alzheimer’s -> Alzheimers
    normalized = normalized.replace("'s", "s").replace("’s", "s")
    normalized = normalized.replace("-", " ").replace("'", " ").replace("’", " ")
    return " ".join(_singularize(token) for token in normalized.split())


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

    @property
    def retryable(self) -> bool:
        fatal_prefixes = ("cui_no_names:", "sapbert_unavailable:")
        return not any(
            reason.startswith(fatal_prefixes) for reason in self.reasons
        )


class SapBertMatcher:
    """Fallback cosine similarity, chỉ load model local khi thật sự cần."""

    def __init__(self, model_path: str | Path | None, threshold: float = 0.85) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.threshold = float(threshold)
        if self.threshold <= 0.0:
            raise ValueError("SapBERT threshold must be > 0")
        self._model = None
        self._error: SapBertUnavailable | None = None

    @property
    def available(self) -> bool:
        if self.model_path is None or not self.model_path.exists():
            return False
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            return False
        return True

    def _load(self):
        if self._error is not None:
            raise self._error
        if self.model_path is None:
            self._error = SapBertUnavailable(
                "no local SapBERT path; pass --sap-model or set SAPBERT_MODEL"
            )
            raise self._error
        if not self.model_path.exists():
            self._error = SapBertUnavailable(
                f"SapBERT path not found: {self.model_path}"
            )
            raise self._error
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            self._error = SapBertUnavailable(
                "sentence-transformers is not installed"
            )
            raise self._error from exc
        try:
            self._model = SentenceTransformer(str(self.model_path))
        except Exception as exc:
            self._error = SapBertUnavailable(f"cannot load SapBERT: {exc}")
            raise self._error from exc
        return self._model

    def score(self, term: str, names: tuple[str, ...]) -> float:
        if not names:
            return -1.0
        model = self._model or self._load()
        vectors = model.encode(
            [term, *names],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        base = vectors[0]
        similarities = [
            sum(float(left) * float(right) for left, right in zip(base, vector))
            for vector in vectors[1:]
        ]
        return max(similarities) if similarities else -1.0


def _exact_cui_match(term: str, names: tuple[str, ...]) -> bool:
    target = canonical_term(term)
    return any(canonical_term(name) == target for name in names)


def validate_translation(
    entities: list[Entity] | tuple[Entity, ...],
    parsed: ParsedSentence,
    hints: dict[str, tuple[str, ...]],
    sap_matcher: SapBertMatcher | None = None,
) -> ValidationResult:
    """Validate toàn bộ tag/span/CUI; trả lý do cụ thể nếu fail."""

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

        names = hints.get(entity.cui, ())
        if not names:
            reasons.append(
                f"cui_no_names: e{entity_index} CUI {entity.cui} has no ENG name"
            )
            continue

        if not _exact_cui_match(en_term, names):
            matched = False
            if sap_matcher is not None and sap_matcher.available:
                try:
                    similarity = sap_matcher.score(en_term, names)
                except SapBertUnavailable as exc:
                    reasons.append(
                        f"sapbert_unavailable: e{entity_index}: {exc}"
                    )
                except Exception as exc:
                    reasons.append(
                        f"sapbert_unavailable: e{entity_index}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                else:
                    if similarity > sap_matcher.threshold:
                        matched = True
                    else:
                        reasons.append(
                            f"cui_mismatch: e{entity_index} {en_term!r} "
                            f"does not match CUI {entity.cui}; "
                            f"SapBERT={similarity:.3f} <= {sap_matcher.threshold}; "
                            f"names={names}"
                        )
            else:
                reasons.append(
                    f"cui_mismatch: e{entity_index} {en_term!r} "
                    f"does not match CUI {entity.cui} names {names}"
                )
            if not matched:
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
