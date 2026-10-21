"""LLM re-ranking (C8) over an already-retrieved candidate list.

The re-ranker never touches the vector index: it only sees candidates, their
display names (supplied by the caller) and the mention/context.
"""

from __future__ import annotations

import math
import re
from typing import Mapping, Protocol, Sequence

from .llm import LLMClient
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


class LLMReranker:
    """Ask the LLM to pick one CUI from the candidate list.

    ``timeout_s`` is forwarded to :meth:`LLMClient.complete`; pass a client
    configured with ``temperature=0.0`` for deterministic choices.  The prompt
    deliberately lists candidates by number and CUI, but only a CUI string is
    accepted as an answer — a bare index such as ``"2"`` is rejected.
    """

    def __init__(self, llm: LLMClient, *, timeout_s: float = 30.0) -> None:
        if isinstance(timeout_s, bool):
            raise ValueError(f"timeout_s must be a number, got {timeout_s!r}")
        try:
            timeout = float(timeout_s)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"timeout_s must be a number, got {timeout_s!r}") from exc
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError(
                f"timeout_s must be a positive finite number, got {timeout_s!r}"
            )
        self._llm = llm
        self._timeout_s = timeout

    def rerank(
        self,
        mention: str,
        context: str,
        candidates: Sequence[Candidate],
        concept_names: Mapping[str, str],
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
        rows = []
        for index, candidate in enumerate(ordered, start=1):
            name = names.get(candidate.cui) if hasattr(names, "get") else None
            if not isinstance(name, str) or not name.strip():
                name = candidate.surface
            rows.append(
                f"{index}. CUI: {candidate.cui} | Name: {name} | "
                f"Surface: {candidate.surface}"
            )

        prompt = (
            f"Mention: {mention}\n"
            f"Context: {context}\n\n"
            "Candidates:\n"
            + "\n".join(rows)
            + "\n\n"
            "Instruction: Select the single candidate that best matches the "
            "mention in context. Answer with exactly one CUI from the list "
            "above. Do not answer with a number, an index, or any extra text."
        )

        try:
            answer = self._llm.complete(prompt, timeout_s=self._timeout_s)
        except Exception as exc:  # surface every client failure as RerankError
            reason = str(exc).strip() or type(exc).__name__
            raise RerankError(
                f"LLM re-rank call failed ({type(exc).__name__}): {reason}"
            ) from exc

        if not isinstance(answer, str) or not answer.strip():
            raise RerankError("LLM re-rank answer was empty or not a string")

        mentioned = {
            candidate.cui
            for candidate in ordered
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
