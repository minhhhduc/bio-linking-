"""Generative relevance feedback (C5) — prompts, parsing and fallback.

Prompt text lives in ``prompts.yaml`` (read by :mod:`GRF.prompts`), not here, so
it can be edited without touching code.  Concurrent LLM failures never
propagate: each one is reported as a
:class:`~GRF.schemas.FeedbackFailure` so the pipeline can degrade to the
original query (architecture §6).
"""

from __future__ import annotations

import json
import os
import re
import threading
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol, Sequence

from .config import FEEDBACK_KINDS
from .llm import LLMClient
from .prompts import digest as prompt_digest
from .prompts import get_prompts
from .schemas import Feedback, FeedbackFailure, Request

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
    """Fill this kind's instruction template from ``prompts.yaml``."""
    prompts = get_prompts()
    if not isinstance(kind, str) or kind not in prompts["feedback"]["instructions"]:
        raise ValueError(f"unknown feedback kind {kind!r}; expected one of {FEEDBACK_KINDS}")
    if isinstance(n_syn, bool) or not isinstance(n_syn, int) or n_syn < 1:
        raise ValueError(f"n_syn must be a positive integer, got {n_syn!r}")
    _require_text("mention", mention)
    _require_text("context", context)
    _require_text("target_kb", target_kb)
    return (
        prompts["feedback"]["instructions"][kind]
        .format(s=context, m=mention, n=n_syn, KB=target_kb)
        .strip()
    )


def build_prompt(mention: str, context: str, instruction: str) -> str:
    """Wrap a filled instruction in the envelope from ``prompts.yaml``."""
    _require_text("mention", mention)
    _require_text("context", context)
    _require_text("instruction", instruction)
    return get_prompts()["feedback"]["envelope"].format(
        context=context, mention=mention, instruction=instruction
    )


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


class JsonlFeedbackCache:
    """Append-only JSONL cache so a run's feedback survives the process.

    One JSON object per line::

        {"ns": "<model>|<prompt digest>", "mention": ..., "context": ...,
         "kind": "def", "texts": ["..."]}

    ``namespace`` pins the entries to the model and prompt template that
    produced them.  Lines written under a different namespace are ignored on
    load and left on disk, so switching model or editing ``prompts.yaml``
    regenerates instead of silently reusing an answer the new prompt would not
    have given.  Reusing the old lines is a matter of setting the namespace
    back.

    Every :meth:`set` appends and flushes, so an interrupted Colab run keeps
    what it already paid for.  The file is opened in append mode and never
    rewritten, so two processes writing the same file cannot truncate it; the
    last line for a key wins on load.
    """

    def __init__(self, path, namespace: str, *, fsync: bool = False) -> None:
        self.path = Path(path)
        if not isinstance(namespace, str) or not namespace.strip():
            raise ValueError("namespace must be a non-empty string")
        self.namespace = namespace
        self._fsync = fsync
        self._items: dict[tuple, tuple[str, ...]] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # A half-written trailing line from a killed run is not a
                    # reason to lose the thousands of good ones before it.
                    continue
                if row.get("ns") != self.namespace:
                    continue
                texts = row.get("texts")
                if not isinstance(texts, list) or not texts:
                    continue
                key = (row.get("mention"), row.get("context"), row.get("kind"))
                if not all(isinstance(part, str) for part in key):
                    continue
                self._items[key] = tuple(texts)

    def get(self, key) -> tuple[str, ...] | None:
        mention, context, kind, _instruction = key
        with self._lock:
            return self._items.get((mention, context, kind))

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

        mention, context, kind, _instruction = key
        record = {
            "ns": self.namespace,
            "mention": mention,
            "context": context,
            "kind": kind,
            "texts": list(value),
        }
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                handle.flush()
                if self._fsync:
                    os.fsync(handle.fileno())
            self._items[(mention, context, kind)] = value

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


class FeedbackGenerator:
    """Generate feedback for whole batches, with per-kind fallback.

    :meth:`generate_batch` is the fast path: every mention's prompts across the
    whole batch are collected, the cache is consulted for all of them, and the
    misses go to the client in one call.  With a local model that is one
    ``generate`` per ``batch_size`` prompts instead of one per prompt, which is
    where the 20,980 s run spent its time.

    :meth:`generate` is the single-mention wrapper, kept for ``link``.
    """

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
        return self.generate_batch(
            [(mention, context, feedback_types, n_syn)], target_kb
        )[0]

    def generate_batch(
        self,
        jobs: Sequence[tuple[str, str, Sequence[str], int]],
        target_kb: str,
    ) -> list[tuple[tuple[Feedback, ...], tuple[FeedbackFailure, ...]]]:
        """One outcome pair per job, in order.

        ``jobs`` holds ``(mention, context, feedback_types, n_syn)``.  A job
        whose kinds are empty yields ``((), ())`` without touching the LLM.
        """
        target_kb = _require_text("target_kb", target_kb)
        prepared = [
            self._prepare(mention, context, kinds, n_syn, target_kb)
            for mention, context, kinds, n_syn in jobs
        ]

        # Everything the cache cannot answer, flattened into one list.
        pending: list[tuple[int, str, str, str, str, int]] = []
        for index, (mention, context, kinds, instructions, n_syn) in enumerate(prepared):
            for kind, instruction in zip(kinds, instructions):
                if self._cache.get((mention, context, kind, instruction)) is None:
                    pending.append((index, mention, context, kind, instruction, n_syn))

        answers: dict[tuple[int, str], str | Exception] = {}
        if pending:
            prompts = [
                build_prompt(mention, context, instruction)
                for _, mention, context, _, instruction, _ in pending
            ]
            try:
                raw_answers: list = list(self._complete_all(prompts))
            except Exception as exc:
                # A batch that failed as a whole (OOM, unloaded model) must not
                # sink the request: every pending item degrades, as a per-call
                # failure would have.
                raw_answers = [exc] * len(pending)
            for (index, _, _, kind, _, _), raw in zip(pending, raw_answers):
                answers[(index, kind)] = raw

        outcomes = []
        for index, (mention, context, kinds, instructions, n_syn) in enumerate(prepared):
            feedbacks: list[Feedback] = []
            failures: list[FeedbackFailure] = []
            for kind, instruction in zip(kinds, instructions):
                key = (mention, context, kind, instruction)
                try:
                    cached = self._cache.get(key)
                    if cached is not None:
                        feedbacks.append(Feedback(kind=kind, texts=tuple(cached)))
                        continue
                    raw = answers[(index, kind)]
                    if isinstance(raw, Exception):
                        raise raw
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
                    feedbacks.append(Feedback(kind=kind, texts=texts))
                except Exception as exc:  # per-kind degradation; never BaseException
                    reason = str(exc).strip()
                    reason = (
                        f"{type(exc).__name__}: {reason}" if reason
                        else type(exc).__name__
                    )
                    failures.append(FeedbackFailure(kind=kind, reason=reason))
            outcomes.append((tuple(feedbacks), tuple(failures)))
        return outcomes

    def _complete_all(self, prompts: Sequence[str]) -> list[str]:
        """Answer every prompt, batched when the client supports it.

        A local model holds a lock across each ``generate``, so many concurrent
        ``complete`` calls would only queue — one ``complete_batch`` is strictly
        better.  A networked client has no such limit, so it keeps the thread
        pool.
        """
        batch = getattr(self._llm, "complete_batch", None)
        if batch is not None:
            return list(batch(prompts))
        # ponytail: threads for HTTP clients only; a local model takes the
        # complete_batch branch above.
        with ThreadPoolExecutor(max_workers=self._max_workers) as pool:
            return list(pool.map(self._llm.complete, prompts))

    @staticmethod
    def _prepare(mention, context, feedback_types, n_syn, target_kb):
        """Validate one job and fill its instructions."""
        mention = _require_text("mention", mention)
        context = _require_text("context", context)
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
        instructions = [
            build_instruction(kind, mention, context, n_syn, target_kb)
            for kind in kinds
        ]
        return mention, context, kinds, instructions, n_syn


def run_feedback_batch(
    generator,
    requests: Sequence,
    target_kb: str,
    *,
    progress_callback=None,
) -> list[tuple[tuple[Feedback, ...], tuple[FeedbackFailure, ...]]]:
    """Phase 2 alone for a whole batch: the feedback text of every request.

    Needs no encoder, no index and no KB — it is the LLM-only half of the
    pipeline, so it can run in a notebook of its own and be handed on as a
    file: :func:`save_feedback_outcomes` writes the result,
    ``EntityLinker.link_batch(..., feedback_outcomes=...)`` replays it.  The
    return value is exactly what ``link_batch`` accepts.

    ``progress_callback(phase, done, total)`` is called with ``"feedback"``
    once per mention.  A local model decodes the whole batch in one
    ``generate`` call, so the counts arrive together at the end.
    """
    if progress_callback is not None and not callable(progress_callback):
        raise ValueError("progress_callback must be callable or None")
    items = list(requests)
    if not items:
        return []
    jobs = [
        (item.mention, item.context, item.feedback_types, item.n_syn)
        for item in items
    ]
    outcomes = generator.generate_batch(jobs, target_kb)
    if progress_callback is not None:
        for done in range(1, len(outcomes) + 1):
            progress_callback("feedback", done, len(outcomes))
    return outcomes


FEEDBACK_BUNDLE_FORMAT = "grf-feedback-v1"


def save_feedback_outcomes(
    path,
    requests: Sequence,
    outcomes,
    *,
    namespace: str,
    progress_callback=None,
) -> None:
    """Write one JSONL record per mention: the request and its feedback.

    This is the handoff between the notebook that runs the LLM and the notebook
    that runs retrieval, so the retrieval pass never needs the feedback model's
    weights.  One record per mention, in request order::

        {"format": "grf-feedback-v1", "ns": "<model>|<prompt digest>",
         "mention": ..., "context": ..., "start_token": ..., "end_token": ...,
         "feedback": [{"kind": "def", "texts": ["..."]}],
         "failures": [{"kind": "def", "reason": "..."}]}

    ``namespace`` pins the file to the model and prompt template that produced
    it; :func:`load_feedback_outcomes` refuses a file from another namespace, so
    a run can never attribute an old answer to a new prompt.
    """
    if not isinstance(namespace, str) or not namespace.strip():
        raise ValueError("namespace must be a non-empty string")
    request_list = list(requests)
    outcome_list = list(outcomes)
    if len(request_list) != len(outcome_list):
        raise ValueError(
            f"got {len(request_list)} requests but {len(outcome_list)} outcomes"
        )

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for done, (request, outcome) in enumerate(
            zip(request_list, outcome_list), start=1
        ):
            feedbacks, failures = outcome
            record = {
                "format": FEEDBACK_BUNDLE_FORMAT,
                "ns": namespace,
                "mention": request.mention,
                "context": request.context,
                "start_token": request.start_token,
                "end_token": request.end_token,
                "feedback": [
                    {"kind": feedback.kind, "texts": list(feedback.texts)}
                    for feedback in feedbacks
                ],
                "failures": [
                    {"kind": failure.kind, "reason": failure.reason}
                    for failure in failures
                ],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            if progress_callback is not None:
                progress_callback(done, len(request_list))


def feedback_namespace(model: str) -> str:
    """The cache/bundle namespace of one feedback model: ``<model>|<digest>``.

    One function for both notebooks: the bundle written by the feedback
    notebook and the one the ranking notebook accepts are pinned by the same
    string, so a mismatch is a message, not a silently replayed old answer.
    """
    return f"{_require_text('model', model)}|{prompt_digest()}"


def load_feedback_outcomes(path, *, namespace: str | None = None,
                           progress_callback=None):
    """Read a file written by :func:`save_feedback_outcomes`.

    Returns ``(requests, outcomes)``: the requests rebuilt from the file (so the
    retrieval notebook needs neither the gold loader nor the KB), and the
    ``(feedbacks, failures)`` pairs ``link_batch`` accepts as
    ``feedback_outcomes``.  A record from another ``namespace`` raises instead
    of being silently reused.

    ``progress_callback(done, total)`` counts records, like the other loaders.
    """
    if progress_callback is not None and not callable(progress_callback):
        raise ValueError("progress_callback must be callable or None")
    source = Path(path)
    requests: list[Request] = []
    outcomes: list[tuple[tuple[Feedback, ...], tuple[FeedbackFailure, ...]]] = []
    namespaces: set[str] = set()

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
            if record.get("format") != FEEDBACK_BUNDLE_FORMAT:
                raise ValueError(
                    f"{where}: unsupported format {record.get('format')!r}; "
                    f"expected {FEEDBACK_BUNDLE_FORMAT!r}"
                )
            namespaces.add(record.get("ns"))

            feedbacks = []
            for index, item in enumerate(record.get("feedback") or []):
                if not isinstance(item, dict):
                    raise ValueError(f"{where}: feedback[{index}] must be an object")
                feedbacks.append(
                    Feedback(kind=item.get("kind"), texts=tuple(item.get("texts") or ()))
                )
            failures = []
            for index, item in enumerate(record.get("failures") or []):
                if not isinstance(item, dict):
                    raise ValueError(f"{where}: failures[{index}] must be an object")
                failures.append(
                    FeedbackFailure(kind=item.get("kind"), reason=item.get("reason"))
                )

            requests.append(
                Request(
                    mention=record.get("mention"),
                    context=record.get("context"),
                    # The feedback is already in this file, so the request asks
                    # for none: retrieval must not need a generator at all.
                    feedback_types=(),
                    start_token=record.get("start_token"),
                    end_token=record.get("end_token"),
                )
            )
            outcomes.append((tuple(feedbacks), tuple(failures)))
            if progress_callback is not None:
                progress_callback(len(requests), None)

    if namespace is not None and namespaces - {namespace}:
        raise ValueError(
            f"{source} was written for {sorted(namespaces - {namespace})}, not "
            f"{namespace!r}; regenerate it with the current model and prompts."
        )
    return requests, outcomes
