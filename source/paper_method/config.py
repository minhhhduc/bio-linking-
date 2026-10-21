"""Pipeline configuration and allowed values — architecture §9."""

from __future__ import annotations

from dataclasses import dataclass

FEEDBACK_KINDS = ("def", "syn", "std")
N_SYN_VALUES = (1, 3, 5, 10)
FUSION_KINDS = ("vector", "rrf")
RERANK_KINDS = ("top1", "llm")


def _check_choice(name: str, value: object, choices: tuple[str, ...]) -> None:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(f"{name} must be one of {choices}, got {value!r}")


def _check_non_empty(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string, got {value!r}")


def _check_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return value


@dataclass(frozen=True)
class PipelineConfig:
    """Immutable runtime configuration for the GRF pipeline.

    Defaults follow architecture §9.  Every invalid value raises ``ValueError``
    from ``__post_init__`` so that a config cannot exist in a half-valid state.
    """

    encoder: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    feedback_types: tuple[str, ...] = ("def",)
    n_syn: int = 5
    target_kb: str = "KB"                 # user-supplied KB name; std only
    top_k_retrieval: int = 10
    top_k_concept: int = 5
    fusion: str = "vector"
    rerank: str = "top1"
    llm_model: str = "gpt-4o"
    index_kb: str | None = None           # optional metadata filter

    def __post_init__(self) -> None:
        _check_non_empty("encoder", self.encoder)

        if not isinstance(self.feedback_types, tuple):
            raise ValueError(
                "feedback_types must be a tuple of feedback kinds, "
                f"got {type(self.feedback_types).__name__}"
            )
        for kind in self.feedback_types:
            if not isinstance(kind, str) or kind not in FEEDBACK_KINDS:
                raise ValueError(
                    f"unknown feedback kind {kind!r}; "
                    f"allowed values are {FEEDBACK_KINDS}"
                )
        if len(set(self.feedback_types)) != len(self.feedback_types):
            raise ValueError(f"duplicate feedback kinds in {self.feedback_types!r}")

        n_syn = _check_int("n_syn", self.n_syn)
        if n_syn not in N_SYN_VALUES:
            raise ValueError(f"n_syn must be one of {N_SYN_VALUES}, got {n_syn}")

        _check_non_empty("target_kb", self.target_kb)

        top_k_retrieval = _check_int("top_k_retrieval", self.top_k_retrieval)
        if top_k_retrieval < 1:
            raise ValueError(
                f"top_k_retrieval must be >= 1, got {top_k_retrieval}"
            )

        top_k_concept = _check_int("top_k_concept", self.top_k_concept)
        if not 1 <= top_k_concept <= 10:
            raise ValueError(
                f"top_k_concept must be in 1..10, got {top_k_concept}"
            )

        _check_choice("fusion", self.fusion, FUSION_KINDS)
        _check_choice("rerank", self.rerank, RERANK_KINDS)
        _check_non_empty("llm_model", self.llm_model)
        if self.index_kb is not None:
            _check_non_empty("index_kb", self.index_kb)
