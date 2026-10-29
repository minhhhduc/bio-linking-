"""LLM re-ranking (C8) over an already-retrieved candidate list.

The re-ranker never touches the vector index: it only sees candidates, their
display names (supplied by the caller) and the mention/context.
"""

from __future__ import annotations

import math
import re
from typing import Mapping, Protocol, Sequence

from .llm import LLMClient
from .prompts import get_prompts
from .schemas import Candidate


class RerankError(RuntimeError):
    """Raised when the LLM answer cannot be mapped to exactly one candidate."""


class Reranker(Protocol):
    def rerank(
        self,
        mention: str,
        context: str,
        candidates: Sequence[Candidate],
        concept_names: Mapping[str, str],
    ) -> str: ...


class PickTopOne:
    """Heuristic re-ranker: keep the highest-scoring candidate (architecture §8)."""

    def rerank(
        self,
        mention: str,
        context: str,
        candidates: Sequence[Candidate],
        concept_names: Mapping[str, str],
    ) -> str:
        ordered = tuple(candidates)
        if not ordered:
            raise ValueError("candidates must not be empty")

        scored: list[tuple[float, str]] = []
        for candidate in ordered:
            if not isinstance(candidate.cui, str) or not candidate.cui.strip():
                raise ValueError("every candidate must have a non-empty CUI")
            if isinstance(candidate.score, bool):
                raise ValueError(
                    f"candidate {candidate.cui!r} has a boolean score"
                )
            try:
                score = float(candidate.score)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"candidate {candidate.cui!r} has a non-numeric score"
                ) from exc
            if not math.isfinite(score):
                raise ValueError(
                    f"candidate {candidate.cui!r} has a non-finite score"
                )
            scored.append((score, candidate.cui))

        # Architecture §8: pick the largest Score(c), with the same
        # deterministic CUI tie-break used by concept aggregation.
        return min(scored, key=lambda item: (-item[0], item[1]))[1]


def select_with_fallback(
    reranker: Reranker,
    mention: str,
    context: str,
    candidates: Sequence[Candidate],
    concept_names: Mapping[str, str] | None = None,
) -> tuple[str, str | None]:
    """Run ``reranker`` and fall back to Top-1 on ``RerankError``.

    Returns ``(cui, error)``; ``error`` is ``None`` on success.  Shared by the
    live pipeline and the end-to-end notebook that re-ranks saved candidates.
    """
    ordered = tuple(candidates)
    if not ordered:
        raise ValueError("candidates must not be empty")
    names = concept_names if concept_names is not None else {}
    try:
        cui = reranker.rerank(mention, context, ordered, names)
        if not isinstance(cui, str) or cui not in {c.cui for c in ordered}:
            raise RerankError(
                f"re-ranker returned {cui!r}, which is not a candidate"
            )
    except RerankError as exc:
        return ordered[0].cui, str(exc)
    return cui, None


def _require_text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string, got {value!r}")
    return value


def _cui_pattern(cui: str) -> re.Pattern[str]:
    # A CUI must stand alone: not glued to an alphanumeric run, and not the
    # suffix of another namespaced id (e.g. "0001" inside "CUI:0001").
    return re.compile(
        r"(?<![A-Za-z0-9_:])" + re.escape(cui) + r"(?![A-Za-z0-9_])"
    )


def _parse_answer(answer: object, candidates: Sequence[Candidate]) -> str:
    """Map one answer to exactly one candidate CUI, or raise ``RerankError``."""
    if not isinstance(answer, str) or not answer.strip():
        raise RerankError("LLM re-rank answer was empty or not a string")

    mentioned = {
        candidate.cui
        for candidate in candidates
        if _cui_pattern(candidate.cui).search(answer)
    }
    if not mentioned:
        raise RerankError(
            "LLM re-rank answer did not contain any candidate CUI "
            f"(answer: {answer.strip()[:200]!r})"
        )
    if len(mentioned) > 1:
        raise RerankError(
            f"ambiguous re-rank answer mentions multiple CUIs: {sorted(mentioned)}"
        )
    return mentioned.pop()


class LLMReranker:
    """Ask the LLM to pick one CUI from the candidate list.

    ``timeout_s`` is forwarded to :meth:`LLMClient.complete`; pass a client
    configured with ``temperature=0.0`` for deterministic choices.  ``None``
    means "use the client's own timeout policy": the OpenAI client then applies
    its configured default, while a locally loaded model has none.  The prompt
    deliberately lists candidates by number and CUI, but
    only a CUI string is accepted as an answer — a bare index such as ``"2"``
    is rejected.
    """

    def __init__(self, llm: LLMClient, *, timeout_s: float | None = 30.0) -> None:
        if timeout_s is None:
            self._timeout_s = None
        else:
            if isinstance(timeout_s, bool):
                raise ValueError(f"timeout_s must be a number or None, got {timeout_s!r}")
            try:
                timeout = float(timeout_s)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"timeout_s must be a number or None, got {timeout_s!r}"
                ) from exc
            if not math.isfinite(timeout) or timeout <= 0:
                raise ValueError(
                    f"timeout_s must be a positive finite number, got {timeout_s!r}"
                )
            self._timeout_s = timeout
        self._llm = llm

    def rerank(
        self,
        mention: str,
        context: str,
        candidates: Sequence[Candidate],
        concept_names: Mapping[str, str],
    ) -> str:
        prompt = self._build_prompt(mention, context, candidates, concept_names)
        return _parse_answer(self._call(prompt), candidates)

    def rerank_batch(
        self,
        items: Sequence[tuple[str, str, Sequence[Candidate], Mapping[str, str]]],
    ) -> list[str | Exception]:
        """Re-rank every item, decoding all prompts in one batched call.

        ``items`` holds ``(mention, context, candidates, concept_names)``.  One
        entry per item comes back: the chosen CUI, or the exception that item
        raised.  A per-item failure never sinks the batch — an unusable answer
        degrades that mention only.  A batch-level failure (OOM, unloaded
        weights) raises, and the caller falls back to :meth:`rerank`.

        Batching matters here for the same reason as in feedback: the local
        client holds a lock across each ``generate``, so one prompt per call
        pays the per-call overhead once per mention instead of once per batch.
        """
        prompts: list[str | Exception] = []
        for index, (mention, context, candidates, names) in enumerate(items):
            try:
                prompts.append(self._build_prompt(mention, context, candidates, names))
            except Exception as exc:  # one malformed item must not sink the batch
                prompts.append(
                    RerankError(f"re-rank item {index} is unusable ({type(exc).__name__}): {exc}")
                )
        pending = [prompt for prompt in prompts if isinstance(prompt, str)]
        if not pending:
            return list(prompts)

        batch = getattr(self._llm, "complete_batch", None)
        if batch is None:
            answers: list = []
            for prompt in pending:
                try:
                    answers.append(self._call(prompt))
                except Exception as exc:
                    answers.append(exc)
        else:
            try:
                answers = list(batch(pending))
            except Exception as exc:
                raise RerankError(
                    f"LLM re-rank batch failed ({type(exc).__name__}): "
                    f"{str(exc).strip() or type(exc).__name__}"
                ) from exc
            if len(answers) != len(pending):
                raise RerankError(
                    f"LLM re-rank batch returned {len(answers)} answers "
                    f"for {len(pending)} prompts"
                )

        results: list[str | Exception] = []
        answers_iter = iter(answers)
        for (_, _, candidates, _), prompt in zip(items, prompts):
            if isinstance(prompt, Exception):
                results.append(prompt)
                continue
            answer = next(answers_iter)
            if isinstance(answer, Exception):
                results.append(answer)
                continue
            try:
                results.append(_parse_answer(answer, candidates))
            except RerankError as exc:
                results.append(exc)
        return results

    def _build_prompt(
        self,
        mention: str,
        context: str,
        candidates: Sequence[Candidate],
        concept_names: Mapping[str, str] | None,
    ) -> str:
        _require_text("mention", mention)
        _require_text("context", context)
        ordered = tuple(candidates)
        if not ordered:
            raise ValueError("candidates must not be empty")
        cuis = [candidate.cui for candidate in ordered]
        if any(not isinstance(cui, str) or not cui.strip() for cui in cuis):
            raise ValueError("every candidate must have a non-empty CUI")
        if len(set(cuis)) != len(cuis):
            raise ValueError(f"candidate CUIs must be unique, got {cuis!r}")

        names = concept_names or {}
        prompts = get_prompts()
        rows = []
        for index, candidate in enumerate(ordered, start=1):
            name = names.get(candidate.cui) if hasattr(names, "get") else None
            if not isinstance(name, str) or not name.strip():
                name = candidate.surface
            rows.append(
                prompts["rerank"]["candidate_row"].format(
                    index=index,
                    cui=candidate.cui,
                    name=name,
                    surface=candidate.surface,
                )
            )

        return prompts["rerank"]["instruction"].format(
            mention=mention,
            context=context,
            candidates="\n".join(rows),
        )

    def _call(self, prompt: str) -> str:
        try:
            return self._llm.complete(prompt, timeout_s=self._timeout_s)
        except Exception as exc:
            raise RerankError(
                f"LLM re-rank call failed ({type(exc).__name__}): "
                f"{str(exc).strip() or type(exc).__name__}"
            ) from exc
