"""GRF for biomedical entity linking (architecture C1-C9).

See ``README.md`` for the mapping between components, reference equations and
modules.  ``sentence-transformers`` is imported lazily by
:class:`GRF.encoder.SentenceTransformerEncoder`, so importing this
package needs only numpy.
"""

from .aggregation import aggregate_hits, concept_rank_list
from .config import (
    FEEDBACK_KINDS,
    FUSION_KINDS,
    N_SYN_VALUES,
    RERANK_KINDS,
    PipelineConfig,
)
from .data import (
    DataIssue,
    DataValidationError,
    DatasetLoadResult,
    MentionSample,
    load_bel_jsonl,
)
from .encoder import Encoder, SentenceTransformerEncoder, build_encoder
from .evaluator import Metrics, evaluate, evaluate_all, evaluate_by_label, make_requests
from .feedback import (
    FeedbackGenerator,
    InMemoryFeedbackCache,
    build_instruction,
    build_prompt,
    parse_synonyms,
)
from .fusion import component_weights, interpolate_vectors, rrf_scores
from .index import VectorIndex
from .kb import concepts_from_dataset, load_concepts_jsonl, surface_entries
from .llm import LLMClient, LLMError, OpenAICompatibleClient
from .pipeline import EntityLinker
from .rerank import LLMReranker, PickTopOne, RerankError, Reranker
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
    "Candidate",
    "Concept",
    "DataIssue",
    "DataValidationError",
    "DatasetLoadResult",
    "Encoder",
    "EntityLinker",
    "FEEDBACK_KINDS",
    "FUSION_KINDS",
    "Feedback",
    "FeedbackFailure",
    "FeedbackGenerator",
    "InMemoryFeedbackCache",
    "LLMClient",
    "LLMError",
    "LLMReranker",
    "MentionSample",
    "Metrics",
    "N_SYN_VALUES",
    "OpenAICompatibleClient",
    "PickTopOne",
    "PipelineConfig",
    "RERANK_KINDS",
    "RerankError",
    "Reranker",
    "Request",
    "Response",
    "RetrievalHit",
    "SentenceTransformerEncoder",
    "VectorIndex",
    "aggregate_hits",
    "build_encoder",
    "build_instruction",
    "build_prompt",
    "component_weights",
    "concept_rank_list",
    "concepts_from_dataset",
    "evaluate",
    "evaluate_all",
    "evaluate_by_label",
    "interpolate_vectors",
    "load_bel_jsonl",
    "load_concepts_jsonl",
    "make_requests",
    "parse_synonyms",
    "rrf_scores",
    "surface_entries",
]
