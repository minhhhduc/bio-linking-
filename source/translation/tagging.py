"""Gắn tag ``<e{i}>...</e{i}>`` quanh entity tiếng Việt.

Token index nguồn của gold được tính bằng ``str.split()`` (xem
``source/common/data.py``), nên ở đây cũng dùng whitespace token có char offset
từ ``\\S+``. Tag được chèn theo char offset để không làm thay đổi câu nguồn.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

_WS_TOKEN_RE = re.compile(r"\S+")


@dataclass(frozen=True)
class Entity:
    """Một entity trong dòng gold, giữ nguyên metadata gốc."""

    cui: str
    icd10: tuple[str, ...]
    label: str
    vi_term: str
    vi_start_token_idx: int
    vi_end_token_idx: int


@dataclass(frozen=True)
class TaggedSentence:
    """Câu đã gắn tag và char span của từng entity trên câu gốc."""

    sentence: str
    vi_tagged: str
    char_spans: tuple[tuple[int, int], ...]


class TaggingError(ValueError):
    """Span nguồn không dùng được."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _source_tokens(sentence: str) -> list[tuple[str, int, int]]:
    return [(m.group(0), m.start(), m.end()) for m in _WS_TOKEN_RE.finditer(sentence)]


def _resolve_span(
    sentence: str,
    tokens: list[tuple[str, int, int]],
    entity: Entity,
    entity_index: int,
) -> tuple[int, int]:
    start_idx = entity.vi_start_token_idx
    end_idx = entity.vi_end_token_idx

    if not _is_int(start_idx) or not _is_int(end_idx):
        raise TaggingError(
            "bad_span",
            f"entity {entity_index}: token indices must be integers, "
            f"got {start_idx!r} and {end_idx!r}",
        )
    if start_idx < 0 or end_idx < 0:
        raise TaggingError(
            "bad_span",
            f"entity {entity_index}: token indices must be non-negative, "
            f"got ({start_idx}, {end_idx})",
        )
    if end_idx < start_idx:
        raise TaggingError(
            "bad_span",
            f"entity {entity_index}: end token {end_idx} < start token {start_idx}",
        )
    if start_idx >= len(tokens) or end_idx >= len(tokens):
        raise TaggingError(
            "bad_span",
            f"entity {entity_index}: token indices ({start_idx}, {end_idx}) "
            f"outside sentence with {len(tokens)} whitespace tokens",
        )

    char_start = tokens[start_idx][1]
    char_end = tokens[end_idx][2]
    observed = unicodedata.normalize("NFC", sentence[char_start:char_end])
    expected = unicodedata.normalize("NFC", entity.vi_term)

    if observed != expected:
        raise TaggingError(
            "span_mismatch",
            f"entity {entity_index}: source span {observed!r} "
            f"!= vi_term {expected!r}",
        )

    return char_start, char_end


def attach_tags(sentence: str, entities: list[Entity] | tuple[Entity, ...]) -> TaggedSentence:
    """Chèn tag quanh từng entity, giữ nguyên từng ký tự của câu nguồn.

    ``char_spans`` là toạ độ trên ``sentence`` gốc; thứ tự phần tử khớp thứ tự
    entity trong input. Hai mention giống nhau ở hai vị trí khác nhau sẽ có hai
    span khác nhau và hai tag khác nhau.
    """

    sentence = unicodedata.normalize("NFC", sentence)
    if not entities:
        return TaggedSentence(sentence=sentence, vi_tagged=sentence, char_spans=())

    tokens = _source_tokens(sentence)
    resolved: list[tuple[int, int, int]] = []
    for index, entity in enumerate(entities):
        char_start, char_end = _resolve_span(sentence, tokens, entity, index)
        if char_end <= char_start:
            raise TaggingError("empty_span", f"entity {index}: empty source span")
        resolved.append((char_start, char_end, index))

    # Entity không được chồng token nhau. Sắp theo vị trí để phát hiện overlap.
    ordered = sorted(resolved, key=lambda item: (item[0], item[1]))
    for previous, current in zip(ordered, ordered[1:]):
        if current[0] < previous[1]:
            raise TaggingError(
                "overlap",
                f"entities {previous[2]} and {current[2]} overlap "
                f"({previous[0]}, {previous[1]}) vs ({current[0]}, {current[1]})",
            )

    # Chèn từ phải sang trái để offset của các span bên trái không đổi.
    tagged = sentence
    for char_start, char_end, entity_index in sorted(
        resolved, key=lambda item: item[0], reverse=True
    ):
        tagged = (
            tagged[:char_start]
            + f"<e{entity_index}>"
            + tagged[char_start:char_end]
            + f"</e{entity_index}>"
            + tagged[char_end:]
        )

    spans = tuple((item[0], item[1]) for item in resolved)
    return TaggedSentence(sentence=sentence, vi_tagged=tagged, char_spans=spans)
