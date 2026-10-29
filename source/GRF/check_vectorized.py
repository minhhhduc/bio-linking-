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

A third check locks the progress contract: ``link_batch`` reports each phase
under its own name, so one bar can be drawn per phase.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from GRF.config import PipelineConfig  # noqa: E402
from GRF.feedback import build_instruction, build_prompt  # noqa: E402
from GRF.index import VectorIndex  # noqa: E402
from GRF.pipeline import PROGRESS_PHASES, EntityLinker  # noqa: E402
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
    """Deterministic text -> vector, with a call counter.

    ``encode_raw`` returns rows of *different* lengths, like a real encoder, so
    a check can tell a raw component from a normalised one; ``encode`` is the
    normalised view of the same rows.
    """

    dim = 16

    def __init__(self) -> None:
        self.calls = 0

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        return rng.normal(size=_FakeEncoder.dim) * rng.uniform(0.5, 12.0)

    def encode_raw(self, texts):
        self.calls += 1
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self._vector(t) for t in texts]).astype(np.float32)

    def encode(self, texts):
        vectors = self.encode_raw(texts)
        if vectors.shape[0] == 0:
            return vectors
        return (vectors / np.linalg.norm(vectors, axis=1, keepdims=True)).astype(
            np.float32
        )

    def encode_mention_in_context(self, mentions, contexts, spans):
        # Raw on purpose: Q_0 must reach the fusion unnormalised.
        return self.encode_raw([f"{m}|{c}" for m, c in zip(mentions, contexts)])


class _FakeFeedback:
    """def/syn/std feedback; the mention ``boom`` loses its def to an LLM error."""

    def generate(self, mention, context, feedback_types, n_syn, target_kb):
        return self.generate_batch(
            [(mention, context, feedback_types, n_syn)], target_kb
        )[0]

    def generate_batch(self, jobs, target_kb):
        outcomes = []
        for mention, context, feedback_types, n_syn in jobs:
            feedbacks, failures = [], []
            for kind in feedback_types:
                if kind == "def" and mention == "boom":
                    failures.append(FeedbackFailure(kind="def", reason="llm down"))
                elif kind == "syn":
                    feedbacks.append(
                        Feedback(
                            kind="syn",
                            texts=tuple(f"syn {mention} {i}" for i in range(n_syn)),
                        )
                    )
                else:
                    feedbacks.append(
                        Feedback(kind=kind, texts=(f"{kind} of {mention}",))
                    )
            outcomes.append((tuple(feedbacks), tuple(failures)))
        return outcomes


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

    # The RRF fusion runs one dense row block at a time; shrink the block so
    # the batch crosses several boundaries instead of staying in the first one.
    import GRF.pipeline as pipeline_module

    original_block = pipeline_module._RRF_ROW_BLOCK

    for fusion in ("vector", "rrf"):
        for index_kb in (None, "KB1"):
            for block in ((original_block,) if fusion == "vector" else (2, original_block)):
                pipeline_module._RRF_ROW_BLOCK = block
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
                tag = f"fusion={fusion} index_kb={index_kb} block={block}"

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
                    if fusion == "rrf":
                        # The rank lists must survive the row blocking intact.
                        # ``link`` keeps the per-pipeline debug under
                        # ``debug["retrieval"]``; ``link_batch`` does the same.
                        assert (
                            single.debug["retrieval"]["rank_lists"]
                            == batch.debug["retrieval"]["rank_lists"]
                        ), f"{tag} row={row}: rank_lists"
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

                # The batch must actually batch: one encoder call for all
                # mentions plus one for all feedback texts.
                assert batched_encoder.calls == 2, (
                    f"{tag}: {batched_encoder.calls} encoder calls, expected 2"
                )

    pipeline_module._RRF_ROW_BLOCK = original_block


def check_progress() -> None:
    """``link_batch`` reports each phase separately, in ``PROGRESS_PHASES`` order.

    One bar per phase is a contract: the callback carries the phase name, so a
    caller can label it.  A phase that reported under another phase's name (the
    original bug: feedback counted into a "retrieval" bar) fails here.
    """
    rng = np.random.default_rng(11)
    count = 40
    vectors = rng.normal(size=(count, 16)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    index = VectorIndex(
        vectors,
        [f"C{i:03d}" for i in range(count)],
        [f"S{i}" for i in range(count)],
    )
    requests = [
        Request(mention=f"m{i}", context=f"ctx {i}", feedback_types=("def",))
        for i in range(5)
    ]

    for fusion in ("vector", "rrf"):
        config = PipelineConfig(fusion=fusion, top_k_retrieval=5, top_k_concept=3)
        linker = EntityLinker(
            index, _FakeEncoder(), config, feedback_generator=_FakeFeedback()
        )
        events: list[tuple[str, int, int | None]] = []
        linker.link_batch(
            requests, progress_callback=lambda p, d, t: events.append((p, d, t))
        )

        seen = list(dict.fromkeys(phase for phase, _, _ in events))
        # Every phase must report, in PROGRESS_PHASES order.
        assert seen == [p for p in PROGRESS_PHASES if p in seen], (
            f"fusion={fusion}: phases reported out of order: {seen}"
        )
        assert seen == list(PROGRESS_PHASES), (
            f"fusion={fusion}: missing phases, got {seen}"
        )
        # Feedback, aggregation and ranking each count all five mentions.
        for phase in ("feedback", "aggregation", "ranking"):
            counted = [done for name, done, _ in events if name == phase]
            assert counted == list(range(1, 6)), (
                f"fusion={fusion} phase={phase}: {counted}"
            )
        # The retrieval phase counts scans, not mentions.
        assert {t for name, _, t in events if name == "retrieval"} == {1}, (
            f"fusion={fusion}: retrieval reported {events}"
        )


def check_prompts() -> None:
    """prompts.yaml loads, and every template carries its placeholders."""
    from GRF.prompts import digest, get_prompts

    prompts = get_prompts()
    for kind in ("def", "syn", "std"):
        assert kind in prompts["feedback"]["instructions"], kind
    assert len(digest()) == 16, "digest must be a short hex string"

    # An instruction is filled from the file, so editing the YAML changes what
    # the LLM is asked.  Checked against the loaded text, not a hard-coded copy.
    for kind in ("def", "syn", "std"):
        filled = build_instruction(kind, "M", "C", 3, "KB")
        assert "M" in filled, (kind, filled)
    assert "C" in build_prompt("M", "C", "I"), "envelope must carry the context"
    assert "I" in build_prompt("M", "C", "I"), "envelope must carry the instruction"


def check_feedback_cache() -> None:
    """A JSONL cache survives a reload and ignores a stale namespace."""
    import tempfile
    from pathlib import Path

    from GRF.feedback import JsonlFeedbackCache

    path = Path(tempfile.mkdtemp()) / "fb.jsonl"
    key = ("m", "c", "def", "instruction")
    first = JsonlFeedbackCache(path, "model|digest1")
    assert first.get(key) is None
    first.set(key, ("answer",))

    reloaded = JsonlFeedbackCache(path, "model|digest1")
    assert reloaded.get(key) == ("answer",), "cache lost its entry on reload"

    # Editing prompts.yaml or switching model changes the namespace; the old
    # answers must not be reused.
    stale = JsonlFeedbackCache(path, "model|digest2")
    assert stale.get(key) is None, "a stale namespace must not answer"


def check_batched_feedback() -> None:
    """generate_batch asks the client once, and link_batch can skip the LLM."""
    from GRF.feedback import FeedbackGenerator

    class BatchClient:
        def __init__(self) -> None:
            self.sizes: list[int] = []

        def complete(self, prompt, *, timeout_s=None):
            raise AssertionError("a batching client must not be called one by one")

        def complete_batch(self, prompts):
            self.sizes.append(len(prompts))
            return ["ok" for _ in prompts]

    client = BatchClient()
    generator = FeedbackGenerator(client)
    jobs = [(f"m{i}", f"c{i}", ("def", "std"), 5) for i in range(4)]
    outcomes = generator.generate_batch(jobs, "KB")
    assert len(outcomes) == 4
    # Two kinds per job, all misses: one batched call for all eight prompts.
    assert client.sizes == [8], client.sizes
    assert all(fb for fb, _ in outcomes), outcomes

    # Running again is served entirely from cache: no further LLM call.
    generator.generate_batch(jobs, "KB")
    assert client.sizes == [8], "cached feedback must not call the LLM again"

    # link_batch with feedback_outcomes must not touch the generator at all.
    rng = np.random.default_rng(11)
    vectors = rng.normal(size=(40, 16)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    index = VectorIndex(
        vectors, [f"C{i:03d}" for i in range(40)], [f"S{i}" for i in range(40)]
    )
    requests = [Request(mention=f"m{i}", context=f"c{i}") for i in range(4)]

    class Exploding:
        def generate_batch(self, jobs, target_kb):
            raise AssertionError("the LLM phase must be skipped")

        def generate(self, *args, **kwargs):
            raise AssertionError("the LLM phase must be skipped")

    linker = EntityLinker(
        index,
        _FakeEncoder(),
        PipelineConfig(top_k_retrieval=5, top_k_concept=3),
        feedback_generator=Exploding(),
    )
    responses = linker.link_batch(requests, feedback_outcomes=outcomes)
    assert len(responses) == 4, responses

    # Two passes — feedback first, then retrieval — must produce exactly what
    # one combined pass produces, and must not re-report the feedback phase.
    requested = [
        Request(mention=f"m{i}", context=f"c{i}", feedback_types=("def", "std"))
        for i in range(4)
    ]
    config = PipelineConfig(
        top_k_retrieval=5, top_k_concept=3, feedback_types=("def", "std")
    )
    together = EntityLinker(
        index, _FakeEncoder(), config, feedback_generator=_FakeFeedback()
    )
    combined = together.link_batch(requested)
    split_linker = EntityLinker(
        index, _FakeEncoder(), config, feedback_generator=_FakeFeedback()
    )
    separate = split_linker.link_batch(
        requested, feedback_outcomes=split_linker.feedback_batch(requested)
    )
    assert [r.cui for r in separate] == [r.cui for r in combined]
    assert [[c.cui for c in r.candidates] for r in separate] == [
        [c.cui for c in r.candidates] for r in combined
    ], "the split pass must rank identically"
    phases: list[str] = []
    split_linker.link_batch(
        requested,
        feedback_outcomes=outcomes,
        progress_callback=lambda phase, done, total: phases.append(phase),
    )
    assert "feedback" not in phases, "replayed feedback must not report that phase"


def check_batched_rerank() -> None:
    """rerank_candidates decodes all prompts at once and keeps per-mention fallback."""
    from common.data import MentionSample

    from GRF.candidates import rerank_candidates
    from GRF.rerank import LLMReranker
    from GRF.schemas import Candidate, Response

    def sample(index: int) -> MentionSample:
        return MentionSample(
            mention=f"m{index}",
            context=f"c{index}",
            gold_cui="C001",
            label=None,
            start_token=None,
            end_token=None,
            record_index=index,
            entity_index=0,
        )

    class BatchClient:
        def __init__(self, answers) -> None:
            self.sizes: list[int] = []
            self._answers = list(answers)

        def complete(self, prompt, *, timeout_s=None):
            raise AssertionError("a batching client must not be called one by one")

        def complete_batch(self, prompts):
            self.sizes.append(len(prompts))
            return list(self._answers[: len(prompts)])

    candidates = (
        Candidate(cui="C001", score=0.9, surface="alpha"),
        Candidate(cui="C002", score=0.8, surface="beta"),
    )
    # Three items; the middle answer names no CUI and must degrade to Top-1.
    client = BatchClient(["C002", "no idea", "C001"])
    reranker = LLMReranker(client, timeout_s=None)

    samples = [sample(i) for i in range(3)]
    responses = [
        Response(cui="C001", candidates=candidates, debug={}) for _ in range(3)
    ]
    reranked = rerank_candidates(samples, responses, reranker=reranker)

    assert client.sizes == [3], client.sizes
    assert [r.cui for r in reranked] == ["C002", "C001", "C001"], [r.cui for r in reranked]
    assert reranked[1].debug["rerank"]["error"], "the failed item must record a reason"
    assert reranked[0].debug["rerank"]["error"] is None

    # A batch-level failure falls back to the per-mention path, which degrades
    # each mention instead of losing the run.
    class Exploding(BatchClient):
        def complete_batch(self, prompts):
            raise RuntimeError("out of memory")

        def complete(self, prompt, *, timeout_s=None):
            return "C002"

    failing = rerank_candidates(
        samples, responses, reranker=LLMReranker(Exploding([]), timeout_s=None)
    )
    assert [r.cui for r in failing] == ["C002"] * 3, [r.cui for r in failing]


def check_single_normalisation() -> None:
    """A retrieval vector is L2-normalised exactly once, after ``Q_GRF``.

    The components of the fusion must arrive raw.  If ``Q_0`` or a ``Q_F`` were
    normalised first, the fused vector would differ from
    ``normalize(raw weighted sum)`` -- and it would silently give the components
    a fixed 1:1 weight instead of the formula's ``1/(n+1)``.
    """
    from GRF.fusion import component_weights, interpolate_matrix, interpolate_vectors

    rng = np.random.default_rng(23)
    q0 = rng.normal(size=16) * 4.0          # raw, not unit
    feedback = [
        rng.normal(size=16) * 2.0,
        rng.normal(size=16) * 0.25,
    ]
    w0, weights = component_weights(len(feedback))

    fused = interpolate_vectors(q0, feedback, (w0, *weights))
    expected = w0 * q0
    for weight, vector in zip(weights, feedback):
        expected = expected + weight * vector
    expected = expected / np.linalg.norm(expected)
    assert np.allclose(fused, expected, atol=1e-6), "fusion must normalise once"
    assert abs(float(np.linalg.norm(fused)) - 1.0) < 1e-6, "Q_GRF must be unit"

    # The matrix path must agree with the per-vector one.
    matrix = interpolate_matrix(
        q0[None, :], [np.stack(feedback)], [np.array([w0, *weights])]
    )
    assert np.allclose(matrix[0], fused, atol=1e-6), "matrix fusion != vectors"

    # Pre-normalising a component changes the result: that is the error guarded
    # against here.  (If it did not change it, this check could not detect it.)
    scaled = np.stack([v / np.linalg.norm(v) for v in feedback])
    pre = interpolate_vectors(q0, list(scaled), (w0, *weights))
    assert not np.allclose(pre, fused, atol=1e-3), (
        "a normalised feedback vector must change Q_GRF"
    )

    # The encoder hands the components back raw; encode() is the normalised
    # view of the same rows, and only the index rows use it.
    encoder = _FakeEncoder()
    raw = encoder.encode_raw(["aspirin", "đau đầu"])
    unit = encoder.encode(["aspirin", "đau đầu"])
    assert not np.allclose(np.linalg.norm(raw, axis=1), 1.0), "encode_raw scaled"
    assert np.allclose(np.linalg.norm(unit, axis=1), 1.0), "encode not unit"
    assert np.allclose(unit, raw / np.linalg.norm(raw, axis=1, keepdims=True))

    # Q_0 comes back raw as well: the mean-pooled mention tokens alone, with no
    # E(m) term added and the one normalisation deferred to the fusion.
    q0_rows = encoder.encode_mention_in_context(
        ["aspirin", "đau đầu"], ["take aspirin", "bị đau đầu"], [(1, 1), (1, 1)]
    )
    assert not np.allclose(np.linalg.norm(q0_rows, axis=1), 1.0), (
        "Q_0 must reach the fusion unnormalised"
    )

    # The index leaves an already-unit query untouched, so the single
    # normalisation the fusion did is not repeated during the cosine scan.
    index = VectorIndex(unit.copy(), ["C1", "C2"], ["aspirin", "đau đầu"])
    from common.index import as_unit_rows

    assert np.array_equal(as_unit_rows(unit), unit), "unit rows were rescaled"
    assert index.search(fused, 1), "search must still work on a unit query"


def main() -> None:
    check_index()
    print("index: OK")
    check_pipeline()
    print("pipeline: OK")
    check_progress()
    print("progress: OK")
    check_prompts()
    print("prompts: OK")
    check_feedback_cache()
    print("feedback cache: OK")
    check_batched_feedback()
    print("batched feedback: OK")
    check_batched_rerank()
    print("batched rerank: OK")
    check_single_normalisation()
    print("single normalisation: OK")
    print("OK")


if __name__ == "__main__":
    main()
