"""Shared retrieval primitives used by GRF and the dense baseline.

This package has no dependency on :mod:`GRF`: it holds the encoder, vector
index, KB/gold loaders, candidate I/O, max-over-aliases aggregation and the
R@1/R@5/accuracy metrics. GRF adds feedback, fusion and re-ranking on top.
"""

from .aggregation import aggregate_hits, concept_rank_list
from .candidates import (
    FORMAT,
    load_candidates,
    save_candidates,
)
from .data import (
    DataIssue,
    DataValidationError,
    DatasetLoadResult,
    MentionSample,
    load_bel_jsonl,
)
from .encoder import Encoder, SentenceTransformerEncoder
from .evaluator import Metrics, accuracy_at_k, metrics_from_responses
from .index import SINGLE_FILE_FORMAT, VectorIndex
from .kb import (
    concepts_from_dataset,
    load_concepts_csv,
    load_concepts_jsonl,
    surface_entries,
)
from .schemas import (
    Candidate,
    Concept,
    Feedback,
    FeedbackFailure,
    Request,
    Response,
    RetrievalHit,
)

__all__ = [
    "FORMAT",
    "Candidate",
    "Concept",
    "DataIssue",
    "DataValidationError",
    "DatasetLoadResult",
    "Encoder",
    "Feedback",
    "FeedbackFailure",
    "MentionSample",
    "Metrics",
    "Request",
    "Response",
    "RetrievalHit",
    "SINGLE_FILE_FORMAT",
    "SentenceTransformerEncoder",
    "VectorIndex",
    "accuracy_at_k",
    "aggregate_hits",
    "concept_rank_list",
    "concepts_from_dataset",
    "load_bel_jsonl",
    "load_candidates",
    "load_concepts_csv",
    "load_concepts_jsonl",
    "metrics_from_responses",
    "save_candidates",
    "surface_entries",
]
