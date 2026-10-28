"""Self-check for the vectorised retrieval phases (architecture §7).

Run from the repository root:

    python source/GRF/check_vectorized.py

Two properties, both release-blocking:

1. ``VectorIndex.search`` / ``search_batch`` return exactly what a full
   ``sorted()`` over the same score matrix returns -- same order, same
   score/cui/surface triples -- including ties, a ``source_kb`` filter,
   ``top_k=1`` and ``top_k`` larger than the eligible row count.
2. ``EntityLinker.link_batch`` matches ``[link(r) for r in requests]`` for both
   fusion modes.  Raw score floats are compared with a tolerance because a
   1-row and an N-row BLAS GEMM reduce in a different order (CONTRACT.md:
   "scores can differ in the last float32 ULPs").
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from GRF.config import PipelineConfig  # noqa: E402
from GRF.index import VectorIndex  # noqa: E402
from GRF.pipeline import EntityLinker  # noqa: E402
from GRF.schemas import Feedback, FeedbackFailure, Request  # noqa: E402

TOLERANCE = 1e-6


def _reference_hits(index, row, top_k, eligible):
    """Full sort over the eligible rows, the behaviour the fast path must match."""
    order = sorted(
        eligible.tolist(),
        key=lambda j: (-float(row[j]), index.cuis[j], index.surfaces[j]),
    )
    return [
        (index.cuis[j], index.surfaces[j], float(row[j]), rank)
        for rank, j in enumerate(order[:top_k], start=1)
    ]


def check_index() -> None:
    rng = np.random.default_rng(7)

    def build(count, surfaces, source_kbs):
        vectors = rng.normal(size=(count, 16)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        cuis = [f"C{i:04d}" for i in range(count)]
        return VectorIndex(vectors, cuis, surfaces, source_kbs)

    count = 500
    cases = {
        "unique surfaces": build(
            count, [f"s {i}" for i in range(count)],
            [None if i % 3 else "kb1" for i in range(count)],
        ),
        "repeated surfaces": build(
            count, [f"s {i % 50}" for i in range(count)],
            [None if i % 3 else "kb1" for i in range(count)],
        ),
        "tiny index": build(20, [f"s {i}" for i in range(20)],
                            [None if i % 3 else "kb1" for i in range(20)]),
        "top_k>size": build(5, [f"s {i}" for i in range(5)],
                            [None if i % 3 else "kb1" for i in range(5)]),
        "no source_kbs": build(count, [f"s {i}" for i in range(count)], None),
        # Every row is the same unit vector: the whole index ties.
        "all ties": VectorIndex(
            np.tile(np.array([[1.0] + [0.0] * 15], dtype=np.float32), (200, 1)),
            [f"C{i:04d}" for i in range(200)],
            [f"s {i % 7}" for i in range(200)],
        ),
    }

    # One query matrix reused across cases; each case scores it with its own
    # rows.  The queries go in un-normalised: the index normalises them itself,
    # and normalising twice in float32 is not idempotent, so the reference must
    # reproduce exactly one normalisation.
    queries = rng.normal(size=(9, 16)).astype(np.float32)
    unit_queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)

    for label, index in cases.items():
        for top_k in ((1, 10, 50) if label == "top_k>size" else (1, 10)):
            scores = unit_queries @ index.vectors.T
            sources = index.source_kbs if index.source_kbs is not None else [None]
            for source_kb in (None, *sorted({s for s in sources if s is not None})):
                eligible = (
                    np.arange(len(index))
                    if source_kb is None
                    else np.array(
                        [j for j, k in enumerate(index.source_kbs) if k == source_kb]
                    )
                )
                if eligible.size == 0:
                    continue
                got = index.search_batch(queries, top_k, source_kb=source_kb)
                for row_number, hits in enumerate(got):
                    expected = _reference_hits(
                        index, scores[row_number], top_k, eligible
                    )
                    actual = [
                        (h.cui, h.surface, h.score, h.rank) for h in hits
                    ]
                    assert actual == expected, (
                        f"{label} top_k={top_k} source_kb={source_kb} "
                        f"row={row_number}\n  got {actual}\n  exp {expected}"
                    )
                # search() must agree with the single-row batch call.  It is the
                # same code path at a different GEMM width, so the raw scores
                # may differ in the last float32 ULP; the ordering may not.
                single = index.search(queries[0], top_k, source_kb=source_kb)
                assert [(h.cui, h.surface, h.rank) for h in single] == [
                    (h.cui, h.surface, h.rank) for h in got[0]
                ], f"{label} top_k={top_k} source_kb={source_kb}: search != batch"
                for left, right in zip(single, got[0]):
                    assert abs(left.score - right.score) <= TOLERANCE, (
                        f"{label} top_k={top_k} source_kb={source_kb}: "
                        f"search score {left.score} vs batch {right.score}"
                    )


class _FakeEncoder:
    """Deterministic text -> unit vector, with a call counter."""

    dim = 16

    def __init__(self) -> None:
        self.calls = 0

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        vector = rng.normal(size=_FakeEncoder.dim)
        return vector / np.linalg.norm(vector)

    def encode(self, texts):
        self.calls += 1
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._vector(t) for t in texts]).astype(np.float32)

    def encode_mention_in_context(self, mentions, contexts, spans):
        return self.encode([f"{m}|{c}" for m, c in zip(mentions, contexts)])


class _FakeFeedback:
    """def/syn/std feedback; the mention ``boom`` loses its def to an LLM error."""

    def generate(self, mention, context, feedback_types, n_syn, target_kb):
        feedbacks, failures = [], []
        for kind in feedback_types:
            if kind == "def" and mention == "boom":
                failures.append(FeedbackFailure(kind="def", reason="llm down"))
            elif kind == "syn":
                feedbacks.append(
                    Feedback(kind="syn", texts=tuple(f"syn {mention} {i}" for i in range(n_syn)))
                )
            else:
                feedbacks.append(Feedback(kind=kind, texts=(f"{kind} of {mention}",)))
        return tuple(feedbacks), tuple(failures)


def check_pipeline() -> None:
    rng = np.random.default_rng(3)
    count = 60
    vectors = rng.normal(size=(count, 16)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    index = VectorIndex(
        vectors,
        [f"C{i:03d}" for i in range(count)],
        [f"S{i}" for i in range(count)],
        [None if i % 4 else "KB1" for i in range(count)],
    )
    requests = [
        Request(mention="aspirin", context="take aspirin daily", feedback_types=("def",)),
        Request(mention="boom", context="boom happens", feedback_types=("def", "syn"), n_syn=3),
        Request(mention="quiet", context="no feedback", feedback_types=()),
        Request(mention="std one", context="ctx", feedback_types=("std",)),
    ]

    for fusion in ("vector", "rrf"):
        for index_kb in (None, "KB1"):
            config = PipelineConfig(
                fusion=fusion,
                top_k_retrieval=7,
                top_k_concept=5,
                index_kb=index_kb,
            )
            one_at_a_time = EntityLinker(
                index, _FakeEncoder(), config, feedback_generator=_FakeFeedback()
            )
            batched_encoder = _FakeEncoder()
            batched = EntityLinker(
                index, batched_encoder, config, feedback_generator=_FakeFeedback()
            )
            singles = [one_at_a_time.link(request) for request in requests]
            together = batched.link_batch(requests)
            tag = f"fusion={fusion} index_kb={index_kb}"

            for row, (single, batch) in enumerate(zip(singles, together)):
                assert single.cui == batch.cui, f"{tag} row={row}: cui"
                assert [c.cui for c in single.candidates] == [
                    c.cui for c in batch.candidates
                ], f"{tag} row={row}: candidate order"
                assert [c.surface for c in single.candidates] == [
                    c.surface for c in batch.candidates
                ], f"{tag} row={row}: candidate surfaces"
                for left, right in zip(single.candidates, batch.candidates):
                    assert abs(left.score - right.score) <= TOLERANCE, (
                        f"{tag} row={row}: score {left.score} vs {right.score}"
                    )
                for key in ("weights", "feedback", "feedback_failures", "score_kind"):
                    assert single.debug.get(key) == batch.debug.get(key), (
                        f"{tag} row={row}: debug[{key}]"
                    )
                assert len(single.debug["concept_ranking"]) == len(
                    batch.debug["concept_ranking"]
                ), f"{tag} row={row}: concept_ranking length"
                for left, right in zip(
                    single.debug["concept_ranking"], batch.debug["concept_ranking"]
                ):
                    assert left["cui"] == right["cui"], f"{tag} row={row}: ranking cui"
                    assert abs(left["score"] - right["score"]) <= TOLERANCE, (
                        f"{tag} row={row}: ranking score"
                    )

            # The batch must actually batch: one encoder call for all mentions
            # plus one for all feedback texts, not one round per request.
            assert batched_encoder.calls == 2, (
                f"{tag}: {batched_encoder.calls} encoder calls, expected 2"
            )


def main() -> None:
    check_index()
    print("index: OK")
    check_pipeline()
    print("pipeline: OK")
    print("OK")


if __name__ == "__main__":
    main()
