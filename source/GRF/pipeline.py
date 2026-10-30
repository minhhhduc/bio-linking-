"""C1-C8 orchestration: vector-based and rank-based GRF pipelines.

Implements architecture §5.2/§5.3 and reference_abstract §5:

* Pipeline 1 (``fusion="vector"``): ``Q_GRF`` is the weighted interpolation of
  ``Q_0`` and the feedback vectors, followed by exactly one index scan.
* ``Q_0`` was formerly ``E(m)``; it now mean-pools the mention's token states
  in its sentence, with the mention kept in place (``common.query.mention_query``
  + ``encoder.encode_mention_in_context``), the same query the dense baseline
  uses.  ``use_context_query=False`` drops the context and uses ``E(m)`` alone,
  like the dense baseline's switch.
* Every component of the fusion (``Q_0``, each ``Q_F``) is **raw**: the encoder
  call that produces it does not L2-normalise.  The retrieval vector is
  normalised exactly once, after ``Q_GRF`` is formed, inside
  ``fusion.interpolate_matrix`` (architecture §4.4, §7.2).
* Pipeline 2 (``fusion="rrf"``): one scan per query component -- ``Q_0`` then
  each ``Q_Fi`` -- so a batch of mentions is scanned ``n+1`` times, once per
  component, each scan holding the same ``b`` rows (architecture §7.3).  The
  ``n+1`` rank matrices are then fused with weighted reciprocal rank fusion.
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
from .feedback import run_feedback_batch
from .fusion import (
    component_weights,
    interpolate_matrix,
    interpolate_vectors,
    rrf_score_matrix,
    rrf_scores,
)
from .rerank import PickTopOne, select_with_fallback
from .schemas import (
    Candidate,
    Feedback,
    FeedbackFailure,
    Request,
    Response,
    RetrievalHit,
)

# Mentions per dense RRF block.  The fusion is a (rows, |concepts|) float64
# matrix; 256 rows keeps that transient near 100 MB for a 50k-concept KB.
_RRF_ROW_BLOCK = 256

# ``link_batch`` progress phases, in execution order.  The unit differs per
# phase, and a caller draws one bar each: mentions for feedback/aggregation/
# ranking, index scans for retrieval (``1`` for vector fusion, ``n+1`` for
# RRF, which scans once per query component), encoder calls (total=None) for
# the two encoding phases.
#
# The last phase is **ranking**, not re-ranking: retrieval stops at the
# ranking, and the saved candidate file is re-ranked later by a separate
# notebook (``reranking.ipynb``).  ``config.rerank == "top1"`` only fills
# ``Response.cui`` with the ranking's first entry, so no re-ranking work runs
# here.
#
# ``"feedback"`` only appears when ``link_batch`` runs that phase itself; a
# caller that supplies ``feedback_outcomes`` -- the two-notebook GRF run, where
# ``feedback.ipynb`` generated the text and unloaded the model -- skips it.
PROGRESS_PHASES = (
    "encode Q0",        # §7.1  encode the query (mention + context)
    "feedback",         # C5    generate F_def / F_syn / F_std
    "encode feedback",  # C3    embed every feedback text
    "retrieval",        # §7.4  cosine matmul + top-k, one scan per Q_x
    "aggregation",      # §7.5  surface hits -> concept scores
    "ranking",          # §7.7  sort by score, cut to top_k_concept
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
        use_context_query: bool = True,
    ) -> None:
        if not isinstance(config, PipelineConfig):
            raise TypeError("config must be a PipelineConfig")
        if not isinstance(use_context_query, bool):
            raise TypeError("use_context_query must be a bool")
        if not (hasattr(index, "search") and hasattr(index, "__len__")):
            raise TypeError("index must provide search() and __len__()")
        if not (hasattr(encoder, "encode") and hasattr(encoder, "dim")):
            raise TypeError("encoder must provide encode() and dim")
        if not hasattr(encoder, "encode_raw"):
            raise TypeError(
                "encoder must provide encode_raw(texts): the fusion components "
                "(Q_0, every Q_F) reach the weighted sum unnormalised, so the "
                "retrieval vector is L2-normalised exactly once, after Q_GRF"
            )
        if not hasattr(encoder, "encode_mention_in_context"):
            raise TypeError(
                "encoder must provide encode_mention_in_context(mentions, "
                "contexts, spans)"
            )

        self._index = index
        self._encoder = encoder
        self.config = config
        self._feedback_generator = feedback_generator
        self._use_context_query = use_context_query

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
    def link(self, request: Request, *, q0: np.ndarray | None = None) -> Response:
        """Link one request. ``q0`` may be precomputed by :meth:`link_batch`."""
        kinds = self._validate_request(request)

        if q0 is None:
            q0 = self._encode_q0([request])[0]
        else:
            q0 = np.asarray(q0, dtype=np.float64)
            if q0.shape != (self._encoder.dim,):
                raise ValueError(
                    f"q0 must have shape ({self._encoder.dim},), got {q0.shape!r}"
                )

        feedbacks, failures = self._generate_feedback(request, kinds)
        specs = self._component_specs(feedbacks)
        # §5 step 2: one component vector per feedback kind, the mean of its
        # texts (a single text for def/std, the n synonyms for syn).
        component_vectors = [
            np.mean(self._encode_raw(texts), axis=0) for _, texts in specs
        ]
        component_labels = [kind for kind, _ in specs]
        w0, feedback_weights = component_weights(len(component_vectors))
        debug = self._fusion_debug(
            q0, specs, component_vectors, (w0, feedback_weights), failures
        )

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
        return self._finish(request, candidates, debug)

    def feedback_batch(
        self,
        requests: Sequence[Request],
        *,
        progress_callback=None,
    ) -> list[tuple[tuple[Feedback, ...], tuple[FeedbackFailure, ...]]]:
        """Phase 2 alone: the feedback text for every request, in order.

        Exists so a caller can run the LLM phase, unload the model, and only
        then load the encoder — the feedback phase is the only one that needs no
        encoder at all, and on a large feedback model the two sets of weights
        do not fit side by side.

        The result can be handed straight to :meth:`link_batch` as
        ``feedback_outcomes``, which then skips phase 2 instead of regenerating.
        """
        if progress_callback is not None and not callable(progress_callback):
            raise ValueError("progress_callback must be callable or None")
        items = list(requests)
        if not items:
            return []
        kinds_per_request = [self._validate_request(item) for item in items]
        return self._run_feedback(
            items, kinds_per_request, progress_callback
        )

    def link_batch(
        self,
        requests: Sequence[Request],
        *,
        feedback_outcomes=None,
        progress_callback=None,
    ) -> list[Response]:
        """Link every request in order, one independent pass per phase.

        Every phase is a batched array operation over the whole request list:

        1. ``Q_0``: one encoder call for all mentions (§4.4).
        2. feedback: one batched LLM call for every prompt (§5).
        3. feedback embeddings: one encoder call for every feedback text.
        4. fusion: ``Q_GRF = W · [Q_0; Q_F]``, a single contraction (vector);
           no vector is formed in the RRF pipeline, only weights.
        5. retrieval: one index scan (vector fusion) or ``n_i + 1`` scans, one
           per query component ``Q_x``, each holding every mention (§7.3).
        6. aggregation: concept scores from the retrieved surface forms (§7.5);
           for RRF, the ``n_i + 1`` rank lists are fused here (§7.6).
        7. ranking: sort each concept ranking, cut to ``top_k_concept`` (§7.7).
           With ``rerank="top1"`` this only copies the ranking's first entry
           into ``Response.cui``; no re-ranking runs in the retrieval stage.

        ``feedback_outcomes`` is what :meth:`feedback_batch` returned.  Supplying
        it skips phase 2, so a caller that already ran the LLM can free its
        weights before any encoder work starts.  Without it, phase 2 runs here
        and the ``"feedback"`` phase is reported as usual.

        ``progress_callback(phase, done, total)``, when supplied, is called
        by each phase with its own unit, so a caller can draw one bar per
        phase (``PROGRESS_PHASES`` lists them in order).
        """
        if progress_callback is not None and not callable(progress_callback):
            raise ValueError("progress_callback must be callable or None")
        items = list(requests)
        if not items:
            return []
        total = len(items)
        kinds_per_request = [self._validate_request(item) for item in items]

        if feedback_outcomes is None:
            # Phase 1 — Q_0 for the whole batch in one encoder call.
            self._report(progress_callback, "encode Q0", 0, None)
            q0_matrix = self._encode_q0(items)
            self._report(progress_callback, "encode Q0", 1, None)
            # Phase 2 — feedback text per mention.
            outcomes = self._run_feedback(
                items, kinds_per_request, progress_callback
            )
        else:
            outcomes = list(feedback_outcomes)
            if len(outcomes) != total:
                raise ValueError(
                    f"feedback_outcomes has {len(outcomes)} entries for "
                    f"{total} requests"
                )
            self._report(progress_callback, "encode Q0", 0, None)
            q0_matrix = self._encode_q0(items)
            self._report(progress_callback, "encode Q0", 1, None)

        feedbacks = [outcome[0] for outcome in outcomes]
        failures = [outcome[1] for outcome in outcomes]

        # Phase 3 — every feedback text of the batch in one encoder call.
        specs_per_request = [
            self._component_specs(row_feedbacks) for row_feedbacks in feedbacks
        ]
        self._report(progress_callback, "encode feedback", 0, None)
        component_vectors = self._encode_components(specs_per_request)
        self._report(progress_callback, "encode feedback", 1, None)

        # Phase 4 — fusion weights, one row per mention (all 1/(n_i+1)).
        weight_rows = [
            component_weights(len(vectors)) for vectors in component_vectors
        ]
        labels_per_request = [
            ["Q_0", *(kind for kind, _ in specs)] for specs in specs_per_request
        ]

        # Phases 5-6 — retrieval and concept aggregation over the batch.
        if self.config.fusion == "vector":
            rankings, retrieval_debugs = self._vector_batch(
                q0_matrix, component_vectors, weight_rows, progress_callback
            )
        else:
            rankings, retrieval_debugs = self._rrf_batch(
                q0_matrix,
                component_vectors,
                labels_per_request,
                weight_rows,
                progress_callback,
            )

        # Phase 7 — assemble responses (re-ranking is the configured stage).
        responses: list[Response] = []
        for done, request in enumerate(items, start=1):
            row = done - 1
            retrieval_debug = retrieval_debugs[row]
            debug = self._fusion_debug(
                q0_matrix[row],
                specs_per_request[row],
                component_vectors[row],
                weight_rows[row],
                failures[row],
            )
            debug["retrieval"] = retrieval_debug
            debug["concept_ranking"] = retrieval_debug.get("concept_ranking", [])
            debug["score_kind"] = retrieval_debug.get("score_kind")
            responses.append(
                self._finish(
                    request,
                    rankings[row][: self.config.top_k_concept],
                    debug,
                )
            )
            self._report(progress_callback, "ranking", done, total)
        return responses

    def _run_feedback(self, items, kinds_per_request, progress_callback) -> list:
        """Phase 2: feedback text for every request, in request order.

        Every prompt of the batch goes to the client at once, so a local model
        decodes one padded batch instead of one prompt at a time.
        """
        if not any(kinds_per_request):
            return [((), ()) for _ in items]
        return run_feedback_batch(
            self._feedback_generator,
            items,
            self.config.target_kb,
            progress_callback=progress_callback,
        )

    @staticmethod
    def _report(progress_callback, phase: str, done: int, total: int | None) -> None:
        """Emit one per-phase progress event; a no-op without a callback."""
        if progress_callback is not None:
            progress_callback(phase, done, total)

    # ------------------------------------------------------------------
    # Batched phases shared by link() and link_batch()
    # ------------------------------------------------------------------
    def _validate_request(self, request: Request) -> tuple[str, ...]:
        """Request checks shared by :meth:`link` and :meth:`link_batch`."""
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
        return kinds

    def _generate_feedback(self, request: Request, kinds: Sequence[str]):
        """One feedback round, validated against the requested kinds."""
        if not kinds:
            return (), ()
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
        return feedbacks, failures

    @staticmethod
    def _component_specs(feedbacks) -> list[tuple[str, list[str]]]:
        """``(kind, texts)`` per successful feedback; def/std hold one text."""
        specs: list[tuple[str, list[str]]] = []
        for feedback in feedbacks:
            if not feedback.texts:
                raise ValueError(f"feedback {feedback.kind!r} has no texts")
            if feedback.kind != "syn" and len(feedback.texts) != 1:
                raise ValueError(
                    f"feedback {feedback.kind!r} must contain exactly one text"
                )
            specs.append((feedback.kind, list(feedback.texts)))
        return specs

    def _encode_components(self, specs_per_request):
        """Mean embedding per component for the whole batch, one encoder call.

        Every component's texts are contiguous in the encoded matrix, so the
        per-component means are one ``np.add.reduceat`` instead of one mean per
        feedback.
        """
        texts: list[str] = []
        starts: list[int] = []
        counts: list[int] = []
        for specs in specs_per_request:
            for _, group in specs:
                starts.append(len(texts))
                counts.append(len(group))
                texts.extend(group)

        vectors_per_request: list[list[np.ndarray]] = [
            [] for _ in specs_per_request
        ]
        if not texts:
            return vectors_per_request
        matrix = self._encode_raw(texts)
        grouped = np.add.reduceat(
            matrix, np.asarray(starts, dtype=np.int64), axis=0
        )
        grouped = grouped / np.asarray(counts, dtype=np.float64)[:, None]
        cursor = 0
        for row, specs in enumerate(specs_per_request):
            for _ in specs:
                vectors_per_request[row].append(grouped[cursor])
                cursor += 1
        return vectors_per_request

    @staticmethod
    def _fusion_debug(
        q0: np.ndarray,
        specs: Sequence[tuple[str, list[str]]],
        component_vectors: Sequence[np.ndarray],
        weight_row: tuple[float, Sequence[float]],
        failures: Sequence[FeedbackFailure],
    ) -> dict:
        w0, feedback_weights = weight_row
        return {
            "Q_0": _as_vector_list(q0),
            "Q_F": {
                kind: _as_vector_list(vector)
                for (kind, _), vector in zip(specs, component_vectors)
            },
            "weights": {
                "Q_0": float(w0),
                **{
                    kind: float(weight)
                    for (kind, _), weight in zip(specs, feedback_weights)
                },
            },
            "feedback": {kind: list(texts) for kind, texts in specs},
            "feedback_failures": [
                {"kind": failure.kind, "reason": failure.reason}
                for failure in failures
            ],
        }

    def _search_rows(self, matrix) -> list[list[RetrievalHit]]:
        """Top-k hits per query row, one batched scan when the index has it."""
        if hasattr(self._index, "search_batch"):
            return self._index.search_batch(
                matrix,
                self.config.top_k_retrieval,
                source_kb=self.config.index_kb,
            )
        return [
            self._index.search(
                row,
                self.config.top_k_retrieval,
                source_kb=self.config.index_kb,
            )
            for row in matrix
        ]

    def _finish(
        self,
        request: Request,
        candidates: Sequence[Candidate],
        debug: dict,
    ) -> Response:
        """Shared tail: apply the configured re-ranker to the candidate list."""
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

    # ------------------------------------------------------------------
    # Batched pipelines (one scan for the whole batch)
    # ------------------------------------------------------------------
    def _vector_batch(
        self,
        q0_matrix: np.ndarray,
        component_vectors: Sequence[Sequence[np.ndarray]],
        weight_rows: Sequence[tuple[float, Sequence[float]]],
        progress_callback=None,
    ) -> tuple[list[list[Candidate]], list[dict]]:
        """Pipeline 1 for the whole batch: one fusion, one index scan.

        Fusion is a single contraction ``Q_GRF = W · [Q_0; Q_F]`` and the
        retrieval is a single ``search_batch`` call, so the batch costs one
        matmul over the index instead of one per mention.

        Reports ``"retrieval"`` (one scan) and ``"aggregation"`` (one row per
        mention) as separate phases.
        """
        rows = len(q0_matrix)
        if len(component_vectors) != rows or len(weight_rows) != rows:
            raise ValueError("component/weight rows must match the batch size")

        blocks = [np.asarray(row, dtype=np.float64) for row in component_vectors]
        scales = [np.asarray([w0, *fw], dtype=np.float64) for w0, fw in weight_rows]
        fused = interpolate_matrix(q0_matrix, blocks, scales)

        self._report(progress_callback, "retrieval", 0, 1)
        hit_lists = self._search_rows(fused)
        self._report(progress_callback, "retrieval", 1, 1)

        rankings: list[list[Candidate]] = []
        debugs: list[dict] = []
        for done, hits in enumerate(hit_lists, start=1):
            ranking = aggregate_hits(hits)
            rankings.append(ranking)
            debugs.append(
                {
                    "fusion": "vector",
                    "score_kind": "cosine",
                    "top_k_retrieval": self.config.top_k_retrieval,
                    "top_k_concept": self.config.top_k_concept,
                    "index_size": len(self._index),
                    "hits": [_hit_debug(hit, "Q_GRF") for hit in hits],
                    "concept_ranking": [_candidate_debug(c) for c in ranking],
                }
            )
            self._report(progress_callback, "aggregation", done, rows)
        return rankings, debugs

    def _rrf_batch(
        self,
        q0_matrix: np.ndarray,
        component_vectors: Sequence[Sequence[np.ndarray]],
        labels_per_request: Sequence[Sequence[str]],
        weight_rows: Sequence[tuple[float, Sequence[float]]],
        progress_callback=None,
    ) -> tuple[list[list[Candidate]], list[dict]]:
        """Pipeline 2 for the whole batch: ``n_i + 1`` scans, then one fusion.

        Queries are laid out **component by component** (architecture §7.3):
        batch 0 is ``Q_0`` over every mention, batch ``i`` is the ``i``-th
        feedback vector of every mention that has one.  Each batch is its own
        ``search_batch`` call, run in order, and each returns its own ranked
        list -- the ``R_x`` the RRF formula sums over.
        """
        rows = len(q0_matrix)
        if len(component_vectors) != rows or len(weight_rows) != rows:
            raise ValueError("component/weight rows must match the batch size")
        if len(labels_per_request) != rows:
            raise ValueError("label rows must match the batch size")
        if rows == 0:
            return [], []

        batches = self._query_batches(
            q0_matrix, component_vectors, labels_per_request, weight_rows
        )
        hits = self._scan_batches(batches, progress_callback)
        return self._rrf_fuse(batches, hits, rows, progress_callback)

    def _query_batches(
        self,
        q0_matrix: np.ndarray,
        component_vectors: Sequence[Sequence[np.ndarray]],
        labels_per_request: Sequence[Sequence[str]],
        weight_rows: Sequence[tuple[float, Sequence[float]]],
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]]:
        """The ``n+1`` query matrices ``Q_x``, one per component (§7.3).

        Each entry is ``(rows, matrix, weights, labels)``: ``matrix[k]`` is the
        query of mention ``rows[k]`` for this component and ``weights[k]`` is
        its fusion weight ``1/(n_j+1)``.  A mention whose ``def`` call failed
        is simply absent from the later batches, so ragged feedback counts need
        no padding -- and one batch holds ``b`` rows instead of ``b(n+1)``.
        """
        rows = len(q0_matrix)
        batches = [
            (
                np.arange(rows),
                np.asarray(q0_matrix, dtype=np.float64),
                np.array([weight_rows[row][0] for row in range(rows)]),
                [labels_per_request[row][0] for row in range(rows)],
            )
        ]

        depth = max((len(vectors) for vectors in component_vectors), default=0)
        for index in range(depth):
            present = [
                row for row in range(rows) if len(component_vectors[row]) > index
            ]
            if not present:
                continue
            if any(
                len(component_vectors[row]) != len(labels_per_request[row]) - 1
                for row in present
            ):
                raise ValueError(
                    "feedback vectors and labels must have equal length"
                )
            batches.append(
                (
                    np.asarray(present, dtype=np.int64),
                    np.stack(
                        [
                            np.asarray(component_vectors[row][index], dtype=np.float64)
                            for row in present
                        ]
                    ),
                    np.array(
                        [weight_rows[row][1][index] for row in present],
                        dtype=np.float64,
                    ),
                    [labels_per_request[row][1 + index] for row in present],
                )
            )
        return batches

    def _scan_batches(self, batches, progress_callback) -> list[list[RetrievalHit]]:
        """One index scan per component batch, in order; the ranked lists ``R_x``."""
        hits: list[list[RetrievalHit]] = []
        total = len(batches)
        self._report(progress_callback, "retrieval", 0, total)
        for done, (_, matrix, _, _) in enumerate(batches, start=1):
            hits.append(self._search_rows(matrix))
            self._report(progress_callback, "retrieval", done, total)
        return hits

    def _rrf_fuse(
        self,
        batches,
        hits_per_batch: Sequence[Sequence[Sequence[RetrievalHit]]],
        rows: int,
        progress_callback,
    ) -> tuple[list[list[Candidate]], list[dict]]:
        """``RRF(c) = Σ_x w_x / r_x(c)`` over the per-component rank matrices.

        The ranks of every component of one mention form the vector
        ``(r_0(c), …, r_n(c))`` of the formula; the fusion is one dense
        scatter-add per row block, ``S[r, c] = Σ_x w_x / r_x(c)``.
        """
        position_of: list[dict[int, int]] = [
            {int(row): position for position, row in enumerate(batch_rows.tolist())}
            for batch_rows, _, _, _ in batches
        ]
        rank_lists = [
            [concept_rank_list(hits) for hits in hits_per_batch[index]]
            for index in range(len(batches))
        ]

        hit_debug: list[list[dict]] = [[] for _ in range(rows)]
        best_surface: list[dict[str, tuple[float, str]]] = [
            {} for _ in range(rows)
        ]
        for index, (batch_rows, _, _, labels) in enumerate(batches):
            for position, (row, label) in enumerate(zip(batch_rows.tolist(), labels)):
                for hit in hits_per_batch[index][position]:
                    hit_debug[row].append(_hit_debug(hit, label))
                    current = best_surface[row].get(hit.cui)
                    if current is None or hit.score > current[0] or (
                        hit.score == current[0] and hit.surface < current[1]
                    ):
                        best_surface[row][hit.cui] = (float(hit.score), hit.surface)

        rankings: list[list[Candidate]] = []
        debugs: list[dict] = []
        # The dense score matrix is (rows, |concepts|); one block at a time
        # bounds that transient for a large gold set.
        # ponytail: fixed block size, not tuned; lower it if the concept
        # vocabulary per block grows past what a Colab runtime can hold.
        for block_start in range(0, rows, _RRF_ROW_BLOCK):
            block_end = min(block_start + _RRF_ROW_BLOCK, rows)
            block_rows = block_end - block_start

            concept_ids: dict[str, int] = {}
            id_rows: list[list[int]] = []
            rank_rows: list[list[int]] = []
            block_owner: list[int] = []
            block_scale: list[float] = []
            block_rank_lists: list[dict[str, dict[str, int]]] = [
                {} for _ in range(block_rows)
            ]
            for index, (_, _, weights, labels) in enumerate(batches):
                for row in range(block_start, block_end):
                    position = position_of[index].get(row)
                    if position is None:
                        continue
                    component_ranks = rank_lists[index][position]
                    if not component_ranks:
                        continue
                    block_rank_lists[row - block_start][labels[position]] = dict(
                        component_ranks
                    )
                    ids: list[int] = []
                    ranks: list[int] = []
                    for cui, rank in component_ranks.items():
                        column = concept_ids.get(cui)
                        if column is None:
                            column = len(concept_ids)
                            concept_ids[cui] = column
                        ids.append(column)
                        ranks.append(rank)
                    id_rows.append(ids)
                    rank_rows.append(ranks)
                    block_owner.append(row - block_start)
                    block_scale.append(float(weights[position]))

            width = max((len(ids) for ids in id_rows), default=0)
            cui_ids = np.full((len(id_rows), width), -1, dtype=np.int64)
            rank_array = np.zeros((len(id_rows), width), dtype=np.float64)
            for index, (ids, ranks) in enumerate(zip(id_rows, rank_rows)):
                if not ids:
                    continue
                cui_ids[index, : len(ids)] = ids
                rank_array[index, : len(ranks)] = ranks

            dense = rrf_score_matrix(
                cui_ids,
                rank_array,
                np.asarray(block_scale, dtype=np.float64),
                np.asarray(block_owner, dtype=np.int64),
                block_rows,
                max(len(concept_ids), 1),
            )

            # Back to CUIs, then the documented ordering: score desc, cui asc.
            inverse = {column: cui for cui, column in concept_ids.items()}
            for row in range(block_start, block_end):
                scores = dense[row - block_start]
                columns = np.flatnonzero(scores)
                order = sorted(
                    columns.tolist(),
                    key=lambda column: (
                        -float(scores[column]),
                        inverse[column],
                    ),
                )
                ranking = [
                    Candidate(
                        cui=inverse[column],
                        score=float(scores[column]),
                        surface=best_surface[row].get(
                            inverse[column], (0.0, inverse[column])
                        )[1],
                    )
                    for column in order
                ]
                rankings.append(ranking)
                debugs.append(
                    {
                        "fusion": "rrf",
                        "score_kind": "rrf",
                        "top_k_retrieval": self.config.top_k_retrieval,
                        "top_k_concept": self.config.top_k_concept,
                        "index_size": len(self._index),
                        "hits": hit_debug[row],
                        "rank_lists": block_rank_lists[row - block_start],
                        "rrf_scores": {
                            candidate.cui: float(candidate.score)
                            for candidate in ranking
                        },
                        "concept_ranking": [_candidate_debug(c) for c in ranking],
                    }
                )
                self._report(progress_callback, "aggregation", len(rankings), rows)
        return rankings, debugs

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
        """Pipeline 2 for one mention, written the long way.

        The readable form of ``_rrf_batch``: one scan per component, in order,
        one rank list per scan, then :func:`fusion.rrf_scores` as a dict.  The
        release check asserts this and the batched path agree.
        """
        if len(feedback_vectors) != len(feedback_labels):
            raise ValueError("feedback vectors and labels must have equal length")

        labels = ["Q_0", *feedback_labels]
        queries = [q0, *feedback_vectors]
        hit_lists = [
            self._search_rows(np.asarray(query, dtype=np.float64)[None, :])[0]
            for query in queries
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
    def _encode_raw(self, texts: Sequence[str]) -> np.ndarray:
        """``E(·)`` unnormalised: the ``E(F_j)`` that ``Q_F`` is a mean of.

        Deliberately *not* ``encoder.encode``: that L2-normalises every row, and
        a scaled feedback vector would enter the weighted sum with a different
        weight than the formula gives it.  The retrieval vector is normalised
        exactly once, after ``Q_GRF`` (``fusion.interpolate_matrix``).
        """
        if isinstance(texts, str):
            raise ValueError("texts must be a sequence of strings, not a string")
        items = list(texts)
        if not items:
            raise ValueError("texts must not be empty")
        for index, text in enumerate(items):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"texts[{index}] must be a non-empty string")

        raw = getattr(self._encoder, "encode_raw", None)
        if raw is None:
            raise ValueError(
                "the encoder must provide encode_raw(texts) so the feedback "
                "vectors enter the fusion unnormalised; only the KB index rows "
                "and a stand-alone query use the normalised encode()"
            )
        matrix = np.asarray(raw(items), dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != len(items):
            raise ValueError(
                "encoder must return an (n, dim) matrix, got "
                f"{matrix.shape!r} for {len(items)} texts"
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError("encoder returned non-finite values")
        return matrix

    def _encode_q0(self, items: Sequence[Request]) -> np.ndarray:
        """Batched ``Q_0``, returned raw (never L2-normalised).

        ``use_context_query=False`` encodes ``mention`` on its own through
        ``encoder.encode_raw`` — the same function the index rows were built
        with, before that function's normalisation.
        """
        if not items:
            return np.zeros((0, self._encoder.dim), dtype=np.float64)
        if self._use_context_query:
            raw = self._encoder.encode_mention_in_context(
                [item.mention for item in items],
                [item.context for item in items],
                [(item.start_token, item.end_token) for item in items],
            )
        else:
            encode_raw = getattr(self._encoder, "encode_raw", None)
            if encode_raw is None:
                raise ValueError(
                    "use_context_query=False needs encode_raw(mentions) so Q_0 "
                    "reaches the fusion unnormalised"
                )
            raw = encode_raw([item.mention for item in items])
        matrix = np.asarray(raw, dtype=np.float64)
        if matrix.shape != (len(items), self._encoder.dim):
            raise ValueError(
                f"encoder must return an ({len(items)}, dim) matrix, got "
                f"{matrix.shape!r}"
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
