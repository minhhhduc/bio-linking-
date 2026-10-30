"""GRF for biomedical entity linking (architecture C1-C9).

See ``README.md`` for the mapping between components, reference equations and
modules.  ``sentence-transformers`` is imported lazily by
:class:`GRF.encoder.SentenceTransformerEncoder`, so importing this
package needs only numpy.
"""

from .aggregation import aggregate_hits, concept_rank_list
from .candidates import load_candidates, rerank_candidates, save_candidates
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
from .evaluator import (
    Metrics,
    evaluate,
    evaluate_all,
    evaluate_by_label,
    make_requests,
    metrics_from_responses,
)
from .feedback import (
    FEEDBACK_BUNDLE_FORMAT,
    FeedbackGenerator,
    InMemoryFeedbackCache,
    build_instruction,
    build_prompt,
    feedback_namespace,
    load_feedback_outcomes,
    parse_synonyms,
    run_feedback_batch,
    save_feedback_outcomes,
)
from .fusion import component_weights, interpolate_vectors, rrf_scores
from .hf_llm import TransformersChatClient
from .index import VectorIndex
from .kb import (
    concepts_from_dataset,
    load_concepts_csv,
    load_concepts_jsonl,
    surface_entries,
)
from .llm import LLMClient, LLMError, OpenAICompatibleClient
from .pipeline import EntityLinker
from .rerank import (
    LLMReranker,
    PickTopOne,
    RerankError,
    Reranker,
    select_with_fallback,
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
    "aggregate_hits",
    "build_encoder",
    "build_instruction",
    "build_prompt",
    "Candidate",
    "component_weights",
    "Concept",
    "concept_rank_list",
    "concepts_from_dataset",
    "DataIssue",
    "DatasetLoadResult",
    "DataValidationError",
    "Encoder",
    "EntityLinker",
    "evaluate",
    "evaluate_all",
    "evaluate_by_label",
    "Feedback",
    "FEEDBACK_BUNDLE_FORMAT",
    "FEEDBACK_KINDS",
    "FeedbackFailure",
    "FeedbackGenerator",
    "feedback_namespace",
    "FUSION_KINDS",
    "InMemoryFeedbackCache",
    "interpolate_vectors",
    "LLMClient",
    "LLMError",
    "LLMReranker",
    "load_bel_jsonl",
    "load_candidates",
    "load_concepts_csv",
    "load_concepts_jsonl",
    "load_feedback_outcomes",
    "make_requests",
    "MentionSample",
    "Metrics",
    "metrics_from_responses",
    "N_SYN_VALUES",
    "OpenAICompatibleClient",
    "parse_synonyms",
    "PickTopOne",
    "PipelineConfig",
    "Request",
    "rerank_candidates",
    "RERANK_KINDS",
    "Reranker",
    "RerankError",
    "Response",
    "RetrievalHit",
    "rrf_scores",
    "run_feedback_batch",
    "save_candidates",
    "save_feedback_outcomes",
    "select_with_fallback",
    "SentenceTransformerEncoder",
    "surface_entries",
    "TransformersChatClient",
    "VectorIndex",
]
