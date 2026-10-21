"""Frozen data types shared across the pipeline.

Field definitions mirror ``docs/architecture.md`` §4 (data contracts) and
``paper/reference_abstract.md`` §2/§3.  These objects are intentionally dumb:
validation that depends on pipeline configuration lives in :mod:`config`.
"""

from __future__ import annotations

from dataclasses import dataclass

FEEDBACK_KINDS = ("def", "syn", "std")


@dataclass(frozen=True)
class Concept:
    """A knowledge-base record ``(c, N_c, A_c)`` — architecture §4.1.

    ``cui`` is the concept id (for example ``"CUI:0001"``), ``name`` is the
    preferred name :math:`N_c`, and ``aliases`` are the surface forms
    :math:`A_c`.  ``source_kb`` optionally records which user-supplied KB
    supplied the concept, matching the index metadata in §5.1 step 4.
    """

    cui: str
    name: str
    aliases: tuple[str, ...] = ()
    source_kb: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.cui, str) or not self.cui.strip():
            raise ValueError("cui must be a non-empty string")
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        if not isinstance(self.aliases, tuple):
            raise ValueError("aliases must be a tuple of strings")
        for index, alias in enumerate(self.aliases):
            if not isinstance(alias, str) or not alias.strip():
                raise ValueError(
                    f"aliases[{index}] must be a non-empty string"
                )
        if self.source_kb is not None and (
            not isinstance(self.source_kb, str) or not self.source_kb.strip()
        ):
            raise ValueError("source_kb must be None or a non-empty string")


@dataclass(frozen=True)
class RetrievalHit:
    """One vector-index hit ``(vector, CUI, surface form)`` — architecture §4.2.

    ``rank`` is 1-based within the list returned by a single index search.
    """

    cui: str
    surface: str
    score: float
    rank: int


@dataclass(frozen=True)
class Candidate:
    """A concept-level candidate produced by max aggregation — architecture §7.3.

    ``score`` is :math:`\\max_{a \\in A_c \\cap Top-k} S(a)` and ``surface`` is
    the surface form of the highest-scoring hit for ``cui``.
    """

    cui: str
    score: float
    surface: str


@dataclass(frozen=True)
class Request:
    """One pipeline request ``(m, s, feedback_types, n_syn)`` — architecture §4.3."""

    mention: str
    context: str
    feedback_types: tuple[str, ...] = ("def",)
    n_syn: int = 5

    def __post_init__(self) -> None:
        # Light validation only; PipelineConfig enforces n_syn in {1, 3, 5, 10}.
        if not isinstance(self.mention, str) or not self.mention.strip():
            raise ValueError("mention must be a non-empty string")
        if not isinstance(self.context, str) or not self.context.strip():
            raise ValueError("context must be a non-empty string")
        if isinstance(self.n_syn, bool) or not isinstance(self.n_syn, int):
            raise ValueError("n_syn must be a positive integer")
        if self.n_syn < 1:
            raise ValueError("n_syn must be a positive integer")


@dataclass(frozen=True)
class Response:
    """Pipeline output — architecture §4.3.

    ``debug`` is a free-form dict populated by the orchestrator with the
    vectors, weights, feedback texts, retrieval hits and rerank mode.
    """

    cui: str
    candidates: tuple[Candidate, ...]
    debug: dict


@dataclass(frozen=True)
class Feedback:
    """Successful LLM feedback for one instruction type — architecture §6.

    ``kind`` is ``"def"``, ``"syn"`` or ``"std"``.  ``texts`` holds one text
    for definition/standard-name feedback and ``n`` texts for synonym feedback.
    """

    kind: str
    texts: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or self.kind not in FEEDBACK_KINDS:
            raise ValueError(f"kind must be one of {FEEDBACK_KINDS}")
        if not isinstance(self.texts, tuple):
            raise ValueError("texts must be a tuple of strings")
        if not self.texts:
            raise ValueError("texts must not be empty")
        for index, text in enumerate(self.texts):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"texts[{index}] must be a non-empty string"
                )


@dataclass(frozen=True)
class FeedbackFailure:
    """A feedback call that failed and degraded to the baseline query (§6)."""

    kind: str
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str) or self.kind not in FEEDBACK_KINDS:
            raise ValueError(f"kind must be one of {FEEDBACK_KINDS}")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")
