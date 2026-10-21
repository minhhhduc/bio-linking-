"""Generative relevance feedback (C5) — prompts, parsing and fallback.

Prompt/instruction texts are copied verbatim from
``paper/reference_abstract.md`` §4.  Concurrent LLM failures never propagate:
each one is reported as a :class:`~paper_method.schemas.FeedbackFailure` so the
pipeline can degrade to the original query (architecture §6).
"""

from __future__ import annotations

import re
import threading
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol, Sequence

from .config import FEEDBACK_KINDS
from .llm import LLMClient
from .schemas import Feedback, FeedbackFailure

_INSTRUCTION_TEMPLATES = {
    "def": (
        "Dựa vào ngữ cảnh {s}, hãy đưa ra một định nghĩa ngắn gọn trong một câu "
        "để giải thích chính xác ý nghĩa của thực thể {m}."
    ),
    "syn": (
        "Dựa vào ngữ cảnh {s}, hãy liệt kê đúng {n} từ đồng nghĩa hoặc các cách "
        "diễn đạt tương đương (alternative phrasings) cho thực thể {m}."
    ),
    "std": (
        "Dựa vào ngữ cảnh {s}, hãy tạo ra tên chuẩn học thuật được cộng đồng "
        "công nhận (standard community-recognized name) cho thực thể {m}, tuân "
        "theo hệ danh pháp của {KB}."
    ),
}

# Numbered list markers: "1. term" / "2) term".  The trailing whitespace (or
# end of string) keeps decimal values such as "1.5 mg" intact.
_NUMBER_RE = re.compile(r"^\s*\d+\s*[.)](?:\s+|$)")
_BULLET_RE = re.compile(r"^\s*[-*•]\s*")
_QUOTES = "\"'`"
_PREAMBLE_RE = re.compile(
    r"^\s*(?:(?:sure|certainly|of course|okay)[!.,:]?\s*)?"
    r"(?:here (?:are|is)|the following|below (?:are|is)|synonyms?|"
    r"từ đồng nghĩa|dưới đây(?: là)?|sau đây(?: là)?)[^:\n]*:\s*",
    re.IGNORECASE,
)


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string, got {value!r}")
    return value


def build_instruction(
    kind: str,
    mention: str,
    context: str,
    n_syn: int,
    target_kb: str,
) -> str:
    """Fill the reference-abstract §4 instruction for ``kind``."""
    if not isinstance(kind, str) or kind not in _INSTRUCTION_TEMPLATES:
        raise ValueError(f"unknown feedback kind {kind!r}; expected one of {FEEDBACK_KINDS}")
    if isinstance(n_syn, bool) or not isinstance(n_syn, int) or n_syn < 1:
        raise ValueError(f"n_syn must be a positive integer, got {n_syn!r}")
    _require_text("mention", mention)
    _require_text("context", context)
    _require_text("target_kb", target_kb)
    return _INSTRUCTION_TEMPLATES[kind].format(
        s=context, m=mention, n=n_syn, KB=target_kb
    )


def build_prompt(mention: str, context: str, instruction: str) -> str:
    """The exact prompt template from architecture §6."""
    return f"Context: {context}\nMention: {mention}\nInstruction: {instruction}"


def _clean_synonym(item: str) -> str:
    text = unicodedata.normalize("NFC", item.strip())
    text = _BULLET_RE.sub("", text)
    text = _NUMBER_RE.sub("", text)
    return text.strip().strip(_QUOTES).strip()


def parse_synonyms(text: str, n: int) -> tuple[str, ...]:
    """Parse an LLM synonym answer into at most ``n`` unique surface forms.

    Blank lines are ignored before the single-line/multi-line decision, so a
    trailing newline cannot turn a comma-separated answer into one fused item.
    A leading sentence such as "Here are 3 synonyms:" is discarded, including
    when it shares the line with the list.  Single-line answers are split on
    ``;`` and then on ``,``; in a multi-line answer, comma splitting is applied
    only to explicitly numbered/bulleted lines, so multi-word synonyms such as
    "alpha, beta" on a bare line are preserved.  Multi-word synonyms are never
    split on spaces.  The answer and every extracted synonym are NFC-normalised
    before deduplication so Vietnamese NFC/NFD spellings collapse.
    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"n must be a positive integer, got {n!r}")
    if not isinstance(text, str) or not text.strip():
        return ()

    # Normalise before removing a Vietnamese preamble and before deduplicating.
    text = unicodedata.normalize("NFC", text)
    text = _PREAMBLE_RE.sub("", text, count=1)
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return ()

    # A title-like first line ("Synonyms:") is not a synonym when a list
    # follows it.
    if len(lines) > 1:
        first = lines[0].strip()
        if (
            first.endswith(":")
            and not _NUMBER_RE.match(first)
            and not _BULLET_RE.match(first)
        ):
            lines = lines[1:]
    if not lines:
        return ()

    if len(lines) == 1:
        raw_items: list[str] = []
        for semicolon_part in lines[0].split(";"):
            raw_items.extend(semicolon_part.split(","))
    else:
        raw_items = []
        for line in lines:
            cleaned = _clean_synonym(line)
            if not cleaned:
                continue
            parts = cleaned.split(";")
            if (
                len(parts) == 1
                and "," in cleaned
                and (_NUMBER_RE.match(line) or _BULLET_RE.match(line))
            ):
                parts = cleaned.split(",")
            raw_items.extend(parts)

    results: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        cleaned = _clean_synonym(item)
        if not cleaned:
            continue
        folded = cleaned.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        results.append(cleaned)
        if len(results) == n:
            break
    return tuple(results)


class FeedbackCache(Protocol):
    def get(self, key) -> tuple[str, ...] | None: ...

    def set(self, key, texts) -> None: ...


class InMemoryFeedbackCache:
    """Thread-safe in-process cache keyed by ``(mention, context, kind, instruction)``."""

    def __init__(self) -> None:
        self._items: dict[tuple, tuple[str, ...]] = {}
        self._lock = threading.Lock()

    def get(self, key) -> tuple[str, ...] | None:
        with self._lock:
            return self._items.get(key)

    def set(self, key, texts) -> None:
        if isinstance(texts, str):
            raise ValueError("texts must be a sequence of strings, not a string")
        try:
            value = tuple(texts)
        except TypeError as exc:
            raise ValueError("texts must be a sequence of strings") from exc
        if not value or any(
            not isinstance(text, str) or not text.strip() for text in value
        ):
            raise ValueError("texts must contain non-empty strings")
        with self._lock:
            self._items[key] = value


class FeedbackGenerator:
    """Generate all requested feedback kinds concurrently, with per-kind fallback."""

    def __init__(
        self,
        llm: LLMClient,
        *,
        cache: FeedbackCache | None = None,
        max_workers: int = 3,
    ) -> None:
        if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
            raise ValueError(f"max_workers must be a positive integer, got {max_workers!r}")
        self._llm = llm
        self._cache: FeedbackCache = cache if cache is not None else InMemoryFeedbackCache()
        self._max_workers = max_workers

    def generate(
        self,
        mention: str,
        context: str,
        feedback_types: Sequence[str],
        n_syn: int,
        target_kb: str,
    ) -> tuple[tuple[Feedback, ...], tuple[FeedbackFailure, ...]]:
        mention = _require_text("mention", mention)
        context = _require_text("context", context)
        target_kb = _require_text("target_kb", target_kb)
        if isinstance(n_syn, bool) or not isinstance(n_syn, int) or n_syn < 1:
            raise ValueError(f"n_syn must be a positive integer, got {n_syn!r}")

        if isinstance(feedback_types, str):
            raise ValueError(
                "feedback_types must be a sequence of feedback kinds, not a string"
            )
        try:
            kinds = tuple(feedback_types)
        except TypeError as exc:
            raise ValueError("feedback_types must be a sequence") from exc
        for kind in kinds:
            if not isinstance(kind, str) or kind not in FEEDBACK_KINDS:
                raise ValueError(
                    f"unknown feedback kind {kind!r}; allowed values are {FEEDBACK_KINDS}"
                )
        if len(set(kinds)) != len(kinds):
            raise ValueError(f"feedback_types must not contain duplicates, got {kinds!r}")
        if not kinds:
            return (), ()

        instructions = [
            build_instruction(kind, mention, context, n_syn, target_kb)
            for kind in kinds
        ]
        workers = min(self._max_workers, len(kinds))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    self._generate_one, mention, context, kind, instruction, n_syn
                )
                for kind, instruction in zip(kinds, instructions)
            ]
            # Reading futures in submission order preserves the caller's order.
            outcomes = [future.result() for future in futures]

        feedbacks: list[Feedback] = []
        failures: list[FeedbackFailure] = []
        for outcome in outcomes:
            if isinstance(outcome, FeedbackFailure):
                failures.append(outcome)
            else:
                feedbacks.append(outcome)
        return tuple(feedbacks), tuple(failures)

    def _generate_one(
        self,
        mention: str,
        context: str,
        kind: str,
        instruction: str,
        n_syn: int,
    ) -> Feedback | FeedbackFailure:
        key = (mention, context, kind, instruction)
        try:
            cached = self._cache.get(key)
            if cached is not None:
                if isinstance(cached, (str, bytes)):
                    raise ValueError(
                        "cached feedback must be a sequence of strings, not a string"
                    )
                return Feedback(kind=kind, texts=tuple(cached))

            raw = self._llm.complete(build_prompt(mention, context, instruction))

            if kind == "syn":
                texts = parse_synonyms(raw, n_syn)
                if not texts:
                    raise ValueError("LLM returned no usable synonyms")
            else:
                text = raw.strip() if isinstance(raw, str) else ""
                if not text:
                    raise ValueError("LLM returned an empty response")
                texts = (text,)

            self._cache.set(key, texts)
            return Feedback(kind=kind, texts=texts)
        except Exception as exc:  # per-kind degradation; never BaseException
            reason = str(exc).strip()
            reason = f"{type(exc).__name__}: {reason}" if reason else type(exc).__name__
            return FeedbackFailure(kind=kind, reason=reason)
