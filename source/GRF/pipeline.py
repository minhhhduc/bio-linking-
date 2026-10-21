"""C1-C8 orchestration: vector-based and rank-based GRF pipelines.

Implements architecture §5.2/§5.3 and reference_abstract §5:

* Pipeline 1 (``fusion="vector"``): ``Q_GRF`` is the weighted interpolation of
  ``Q_0`` and the feedback vectors, followed by exactly one index scan.
* Pipeline 2 (``fusion="rrf"``): one scan per query, concept-level rank lists,
  then weighted reciprocal rank fusion.
* ``feedback_types=()`` degrades to plain dense retrieval (architecture §11).

``n`` in the fusion weights is the number of *successful feedback components*
(``def``/``syn``/``std``), not ``n_syn``.  The reference uses ``n`` for both the
synonym count and the number of feedback components; architecture §7/§9 makes
the weighting explicit (one component per feedback type, all weights
``1/(n+1)``), so ``n_syn`` only controls how many synonyms are generated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .aggregation import aggregate_hits, concept_rank_list
from .config import FEEDBACK_KINDS, N_SYN_VALUES, PipelineConfig
from .fusion import component_weights, interpolate_vectors, rrf_scores
from .rerank import PickTopOne, select_with_fallback
from .schemas import (
    Candidate,
    Feedback,
    FeedbackFailure,
    Request,
    Response,
    RetrievalHit,
)


def _as_vector_list(vector: np.ndarray) -> list[float]:
    """JSON-friendly Python floats (debug output)."""
    return [float(value) for value in np.asarray(vector).ravel()]


def _hit_debug(hit: RetrievalHit, query: str) -> dict:
    return {
        "query": query,
        "cui": hit.cui,
        "surface": hit.surface,
        "score": float(hit.score),
        "rank": int(hit.rank),
    }


def _candidate_debug(candidate: Candidate) -> dict:
    return {
        "cui": candidate.cui,
        "score": float(candidate.score),
        "surface": candidate.surface,
    }


class EntityLinker:
    """Orchestrates encoder, feedback, retrieval, fusion and re-ranking."""

    def __init__(
        self,
        index,
        encoder,
        config: PipelineConfig,
        *,
        feedback_generator=None,
        reranker=None,
        concept_names: Mapping[str, str] | None = None,
    ) -> None:
        if not isinstance(config, PipelineConfig):
            raise TypeError("config must be a PipelineConfig")
        if not (hasattr(index, "search") and hasattr(index, "__len__")):
            raise TypeError("index must provide search() and __len__()")
        if not (hasattr(encoder, "encode") and hasattr(encoder, "dim")):
            raise TypeError("encoder must provide encode() and dim")

        self._index = index
        self._encoder = encoder
        self.config = config
        self._feedback_generator = feedback_generator

        if config.index_kb is not None:
            source_kbs = getattr(index, "source_kbs", None)
            if source_kbs is not None and config.index_kb not in source_kbs:
                available = sorted(
                    {value for value in source_kbs if value is not None}
                )
                raise ValueError(
                    f"index_kb {config.index_kb!r} does not occur in the index; "
                    f"available source_kb values are {available!r}"
                )

        if concept_names is None:
            concept_names = {}
        if not isinstance(concept_names, Mapping):
            raise TypeError("concept_names must be a mapping or None")
        self._concept_names: dict[str, str] = {}
        for cui, name in concept_names.items():
            if not isinstance(cui, str) or not cui.strip():
                raise ValueError("concept_names keys must be non-empty CUI strings")
            if not isinstance(name, str) or not name.strip():
                raise ValueError("concept_names values must be non-empty strings")
            self._concept_names[cui] = name

        if config.rerank == "llm":
            if reranker is None:
                raise ValueError("rerank='llm' requires a reranker instance")
            self._reranker = reranker
        else:
            # Pick Top-1 is the documented heuristic; an injected LLM reranker
            # is deliberately ignored when the config selects top-1.
            self._reranker = PickTopOne()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def link(self, request: Request) -> Response:
        if not isinstance(request, Request):
            raise TypeError("request must be a Request")
        kinds = self._validate_kinds(request.feedback_types)
        if request.n_syn not in N_SYN_VALUES:
            raise ValueError(
                f"n_syn must be one of {N_SYN_VALUES}, got {request.n_syn!r}"
            )
        if kinds and self._feedback_generator is None:
            raise ValueError(
                "feedback_types is non-empty but no feedback_generator was supplied"
            )

        q0 = self._encode_many([request.mention])[0]

        feedbacks: tuple[Feedback, ...] = ()
        failures: tuple[FeedbackFailure, ...] = ()
        if kinds:
            feedbacks, failures = self._feedback_generator.generate(
                request.mention,
                request.context,
                kinds,
                request.n_syn,
                self.config.target_kb,
            )
            feedbacks = tuple(feedbacks)
            failures = tuple(failures)
            requested = set(kinds)
            outcome_kinds: list[str] = []
            for feedback in feedbacks:
                if not isinstance(feedback, Feedback):
                    raise ValueError(
                        "feedback generator returned a non-Feedback success"
                    )
                outcome_kinds.append(feedback.kind)
            for failure in failures:
                if not isinstance(failure, FeedbackFailure):
                    raise ValueError(
                        "feedback generator returned a non-FeedbackFailure failure"
                    )
                outcome_kinds.append(failure.kind)
            if len(outcome_kinds) != len(kinds) or set(outcome_kinds) != requested:
                raise ValueError(
                    "feedback generator must return exactly one outcome per "
                    f"requested kind {kinds!r}, got {outcome_kinds!r}"
                )

        component_vectors: list[np.ndarray] = []
        component_labels: list[str] = []
        qf_debug: dict[str, list[float]] = {}
        feedback_texts: dict[str, list[str]] = {}
        for feedback in feedbacks:
            if not feedback.texts:
                raise ValueError(f"feedback {feedback.kind!r} has no texts")
            if feedback.kind == "syn":
                matrix = self._encode_many(feedback.texts)
                # reference §5 step 2: mean of the synonym embeddings.
                vector = np.mean(matrix, axis=0)
            else:
                if len(feedback.texts) != 1:
                    raise ValueError(
                        f"feedback {feedback.kind!r} must contain exactly one text"
                    )
                vector = self._encode_many(feedback.texts)[0]
            component_vectors.append(vector)
            component_labels.append(feedback.kind)
            qf_debug[feedback.kind] = _as_vector_list(vector)
            feedback_texts[feedback.kind] = list(feedback.texts)

        w0, feedback_weights = component_weights(len(component_vectors))
        weights: dict[str, float] = {"Q_0": float(w0)}
        for label, weight in zip(component_labels, feedback_weights):
            weights[label] = float(weight)

        debug: dict = {
            "Q_0": _as_vector_list(q0),
            "Q_F": qf_debug,
            "weights": weights,
            "feedback": feedback_texts,
            "feedback_failures": [
                {"kind": failure.kind, "reason": failure.reason}
                for failure in failures
            ],
        }

        if self.config.fusion == "vector":
            candidates, retrieval_debug = self._vector_pipeline(
                q0, component_vectors, (w0, *feedback_weights)
            )
        else:
            candidates, retrieval_debug = self._rrf_pipeline(
                q0,
                component_vectors,
                component_labels,
                (w0, *feedback_weights),
            )

        debug["retrieval"] = retrieval_debug
        # ``evaluator`` and the documented debug contract read the untruncated
        # ranking from the top level; keep the per-pipeline copy as well.
        debug["concept_ranking"] = retrieval_debug.get("concept_ranking", [])
        debug["score_kind"] = retrieval_debug.get("score_kind")
        rerank_debug: dict = {"mode": self.config.rerank, "error": None}

        if not candidates:
            debug["rerank"] = rerank_debug
            debug["error"] = "no_candidates"
            return Response(cui="", candidates=(), debug=debug)

        # §6 degradation principle: a failed LLM call must not sink the request.
        cui, rerank_error = select_with_fallback(
            self._reranker,
            request.mention,
            request.context,
            candidates,
            self._concept_names,
        )
        rerank_debug["error"] = rerank_error

        debug["rerank"] = rerank_debug
        return Response(cui=cui, candidates=tuple(candidates), debug=debug)

    def link_batch(
        self,
        requests: Sequence[Request],
        *,
        progress_callback=None,
    ) -> list[Response]:
        """Link every request in order.

        ``progress_callback(done, total)``, when supplied, is called after each
        request (used by the Colab notebooks to drive ``tqdm``).
        """
        if progress_callback is not None and not callable(progress_callback):
            raise ValueError("progress_callback must be callable or None")
        items = list(requests)
        responses: list[Response] = []
        for done, request in enumerate(items, start=1):
            responses.append(self.link(request))
            if progress_callback is not None:
                progress_callback(done, len(items))
        return responses

    # ------------------------------------------------------------------
    # Pipelines
    # ------------------------------------------------------------------
    def _vector_pipeline(
        self,
        q0: np.ndarray,
        feedback_vectors: Sequence[np.ndarray],
        weights: Sequence[float],
    ) -> tuple[list[Candidate], dict]:
        fused = interpolate_vectors(q0, feedback_vectors, weights)
        hits = list(
            self._index.search(
                fused,
                self.config.top_k_retrieval,
                source_kb=self.config.index_kb,
            )
        )
        ranking = aggregate_hits(hits)
        candidates = ranking[: self.config.top_k_concept]
        debug = {
            "fusion": "vector",
            "score_kind": "cosine",
            "top_k_retrieval": self.config.top_k_retrieval,
            "top_k_concept": self.config.top_k_concept,
            "index_size": len(self._index),
            "hits": [_hit_debug(hit, "Q_GRF") for hit in hits],
            "concept_ranking": [_candidate_debug(c) for c in ranking],
        }
        return candidates, debug

    def _rrf_pipeline(
        self,
        q0: np.ndarray,
        feedback_vectors: Sequence[np.ndarray],
        feedback_labels: Sequence[str],
        weights: Sequence[float],
    ) -> tuple[list[Candidate], dict]:
        if len(feedback_vectors) != len(feedback_labels):
            raise ValueError("feedback vectors and labels must have equal length")

        labels = ["Q_0", *feedback_labels]
        queries = [q0, *feedback_vectors]
        if hasattr(self._index, "search_batch"):
            hit_lists = self._index.search_batch(
                queries,
                self.config.top_k_retrieval,
                source_kb=self.config.index_kb,
            )
        else:  # custom index with only search()
            hit_lists = [
                self._index.search(
                    vector,
                    self.config.top_k_retrieval,
                    source_kb=self.config.index_kb,
                )
                for vector in queries
            ]
        searches = list(zip(labels, hit_lists))

        hit_debug: list[dict] = []
        best_surface: dict[str, tuple[float, str]] = {}
        for label, hits in searches:
            for hit in hits:
                hit_debug.append(_hit_debug(hit, label))
                current = best_surface.get(hit.cui)
                if current is None or hit.score > current[0] or (
                    hit.score == current[0] and hit.surface < current[1]
                ):
                    best_surface[hit.cui] = (float(hit.score), hit.surface)

        rank_lists = [concept_rank_list(hits) for _, hits in searches]
        scores = rrf_scores(rank_lists, weights)
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))

        ranking = [
            Candidate(
                cui=cui,
                score=float(score),
                surface=best_surface.get(cui, (0.0, cui))[1],
            )
            for cui, score in ordered
        ]
        candidates = ranking[: self.config.top_k_concept]
        debug = {
            "fusion": "rrf",
            "score_kind": "rrf",
            "top_k_retrieval": self.config.top_k_retrieval,
            "top_k_concept": self.config.top_k_concept,
            "index_size": len(self._index),
            "hits": hit_debug,
            "rank_lists": {
                label: dict(ranking) for label, ranking in zip(
                    (label for label, _ in searches), rank_lists
                )
            },
            "rrf_scores": {cui: float(score) for cui, score in scores.items()},
            "concept_ranking": [_candidate_debug(c) for c in ranking],
        }
        return candidates, debug

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _encode_many(self, texts: Sequence[str]) -> np.ndarray:
        if isinstance(texts, str):
            raise ValueError("texts must be a sequence of strings, not a string")
        items = list(texts)
        if not items:
            raise ValueError("texts must not be empty")
        for index, text in enumerate(items):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(
                    f"texts[{index}] must be a non-empty string"
                )
        matrix = np.asarray(self._encoder.encode(items), dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != len(items):
            raise ValueError(
                "encoder must return an (n, dim) matrix, got "
                f"{matrix.shape!r} for {len(items)} texts"
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError("encoder returned non-finite values")
        return matrix

    @staticmethod
    def _validate_kinds(feedback_types) -> tuple[str, ...]:
        if isinstance(feedback_types, str):
            raise ValueError("feedback_types must be a sequence of kinds, not a string")
        try:
            kinds = tuple(feedback_types)
        except TypeError as exc:
            raise ValueError("feedback_types must be a sequence") from exc
        for kind in kinds:
            if kind not in FEEDBACK_KINDS:
                raise ValueError(
                    f"unknown feedback kind {kind!r}; allowed values are "
                    f"{FEEDBACK_KINDS}"
                )
        if len(set(kinds)) != len(kinds):
            raise ValueError(f"duplicate feedback kinds in {kinds!r}")
        return kinds
