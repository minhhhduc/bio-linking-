"""Parse output của model: JSON, tag, câu sạch và token index tiếng Anh."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass

TOKEN_PATTERN = r"\w+|[^\w\s]"
TOKEN_RE = re.compile(TOKEN_PATTERN)
TAG_RE = re.compile(r"</?e(\d+)>")
FENCED_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
THINKING_RE = re.compile(r" thinking.*?</think>", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class TokenizedSentence:
    text: str
    tokens: tuple[str, ...]
    offsets: tuple[tuple[int, int], ...]

    def span_text(self, start: int, end: int) -> str:
        if start < 0 or end < start or end >= len(self.tokens):
            raise TokenSpanError(
                "token_index", f"invalid inclusive token span ({start}, {end})"
            )
        return " ".join(self.tokens[start:end + 1])


class TokenSpanError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class TagParseError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


class ModelOutputError(ValueError):
    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class ParsedTag:
    tag_id: int
    text: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class ParsedSentence:
    clean: str
    tags: tuple[ParsedTag, ...]


def tokenize_en(text: str) -> TokenizedSentence:
    """Tokenize câu EN bằng regex bắt buộc, giữ char offset."""

    matches = list(TOKEN_RE.finditer(text))
    return TokenizedSentence(
        text=text,
        tokens=tuple(m.group(0) for m in matches),
        offsets=tuple((m.start(), m.end()) for m in matches),
    )


def char_span_to_token_span(
    tokenized: TokenizedSentence,
    char_start: int,
    char_end: int,
) -> tuple[int, int]:
    """Đổi char span half-open sang token span inclusive.

    Span phải bắt đầu đúng ở đầu một token và kết thúc đúng ở cuối một token.
    """

    text = tokenized.text
    if (
        isinstance(char_start, bool)
        or not isinstance(char_start, int)
        or isinstance(char_end, bool)
        or not isinstance(char_end, int)
    ):
        raise TokenSpanError("char_span", "char_start/char_end must be integers")
    if char_start < 0 or char_end > len(text) or char_start >= char_end:
        raise TokenSpanError(
            "char_span",
            f"invalid half-open span ({char_start}, {char_end}) "
            f"for text of length {len(text)}",
        )
    if text[char_start].isspace() or text[char_end - 1].isspace():
        raise TokenSpanError(
            "padded_span",
            f"span ({char_start}, {char_end}) has leading/trailing whitespace",
        )

    inside = [
        index
        for index, (start, end) in enumerate(tokenized.offsets)
        if start >= char_start and end <= char_end
    ]
    if not inside:
        raise TokenSpanError(
            "token_boundary",
            f"span ({char_start}, {char_end}) contains no complete token",
        )

    first = inside[0]
    last = inside[-1]
    first_start = tokenized.offsets[first][0]
    last_end = tokenized.offsets[last][1]
    if first_start != char_start:
        raise TokenSpanError(
            "token_boundary",
            f"char span starts at {char_start}, next token starts at {first_start}",
        )
    if last_end != char_end:
        raise TokenSpanError(
            "token_boundary",
            f"char span ends at {char_end}, previous token ends at {last_end}",
        )
    return first, last


def to_token_idx(text: str, char_start: int, char_end: int) -> tuple[int, int]:
    """Tokenize ``text`` và đổi một char span sang token index inclusive."""

    return char_span_to_token_span(tokenize_en(text), char_start, char_end)


def parse_tagged(tagged: str) -> ParsedSentence:
    """Bỏ tag, trả câu sạch và char span của từng tag.

    Chỉ coi ``<eN>`` / ``</eN>`` là markup. Các ký tự ``<`` / ``>`` dùng làm
    toán tử so sánh trong câu vẫn được giữ nguyên.
    """

    tagged = unicodedata.normalize("NFC", tagged)
    parts: list[str] = []
    stack: list[tuple[int, int]] = []
    seen_open: set[int] = set()
    spans: list[tuple[int, int, int]] = []
    clean_len = 0
    position = 0

    for match in TAG_RE.finditer(tagged):
        chunk = tagged[position:match.start()]
        parts.append(chunk)
        clean_len += len(chunk)

        token = match.group(0)
        tag_id = int(match.group(1))
        closing = token.startswith("</")

        if not closing:
            if stack:
                raise TagParseError(
                    "nested_tag",
                    f"tag e{tag_id} opens while e{stack[-1][0]} is still open",
                )
            if tag_id in seen_open:
                raise TagParseError("duplicate_tag", f"tag e{tag_id} opened twice")
            seen_open.add(tag_id)
            stack.append((tag_id, clean_len))
        else:
            if not stack:
                raise TagParseError(
                    "mismatched_close", f"closing </e{tag_id}> without open tag"
                )
            open_id, char_start = stack.pop()
            if open_id != tag_id:
                raise TagParseError(
                    "mismatched_close",
                    f"closing </e{tag_id}> does not match open <e{open_id}>",
                )
            spans.append((tag_id, char_start, clean_len))

        position = match.end()

    tail = tagged[position:]
    parts.append(tail)
    clean_len += len(tail)

    if stack:
        open_ids = ", ".join(f"e{tag_id}" for tag_id, _ in stack)
        raise TagParseError("unclosed_tag", f"unclosed tag(s): {open_ids}")

    clean = "".join(parts)
    if re.search(r"<\s*/?\s*[eE]\s*[-+]?\d+\s*/?>", clean):
        raise TagParseError(
            "malformed_markup", "tag-like markup remains after parsing"
        )

    tags = tuple(
        sorted(
            (
                ParsedTag(tag_id, clean[start:end], start, end)
                for tag_id, start, end in spans
            ),
            key=lambda tag: tag.char_start,
        )
    )
    return ParsedSentence(clean=clean, tags=tags)


def _strip_thinking(text: str) -> str:
    stripped = THINKING_RE.sub("", text)
    # Một số template chỉ để lại phần sau `</think>` nếu model không mở block.
    if "</think>" in stripped and " thinking" not in stripped:
        stripped = stripped.split("</think>", 1)[1]
    return stripped


def _first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        else:
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start:index + 1]
    return None


def parse_model_json(raw: str) -> dict[str, object]:
    """Lấy object JSON từ raw output; lỗi được coi là validation fail."""

    text = _strip_thinking(raw).strip()
    if not text:
        raise ModelOutputError("bad_response", "model returned an empty response")

    candidates = [text]
    fenced = FENCED_RE.search(text)
    if fenced:
        candidates.append(fenced.group(1).strip())
    first_object = _first_json_object(text)
    if first_object is not None:
        candidates.append(first_object)

    payload: object | None = None
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            break
        except (json.JSONDecodeError, TypeError) as exc:
            last_error = exc

    if payload is None:
        raise ModelOutputError(
            "bad_json", f"response is not valid JSON: {last_error}"
        )
    if not isinstance(payload, dict):
        raise ModelOutputError("bad_json", "JSON response must be an object")

    en_tagged = payload.get("en_tagged")
    if not isinstance(en_tagged, str):
        raise ModelOutputError(
            "bad_response", 'JSON response must contain string field "en_tagged"'
        )
    if not en_tagged.strip():
        raise ModelOutputError("bad_response", '"en_tagged" is empty')
    return payload
