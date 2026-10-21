# Implementation contract — paper_method

Frozen interfaces for parallel implementation. Derived from
`docs/architecture.md` and `paper/reference_abstract.md`. Do not change a
signature here without coordinating; other modules import it.

Package root: `source/paper_method/`. Python 3.10+. `numpy` is the only hard
dependency; `sentence-transformers` is imported lazily by the encoder.

## schemas.py

```python
@dataclass(frozen=True)
class Concept:
    cui: str
    name: str
    aliases: tuple[str, ...] = ()
    source_kb: str | None = None

@dataclass(frozen=True)
class RetrievalHit:
    cui: str
    surface: str
    score: float
    rank: int                      # 1-based within the searched list

@dataclass(frozen=True)
class Candidate:
    cui: str
    score: float
    surface: str

@dataclass(frozen=True)
class Request:
    mention: str
    context: str
    feedback_types: tuple[str, ...] = ("def",)
    n_syn: int = 5

@dataclass(frozen=True)
class Response:
    cui: str
    candidates: tuple[Candidate, ...]
    debug: dict

@dataclass(frozen=True)
class Feedback:
    kind: str                       # "def" | "syn" | "std"
    texts: tuple[str, ...]          # 1 text for def/std, n for syn

@dataclass(frozen=True)
class FeedbackFailure:
    kind: str
    reason: str
```

## config.py

```python
FEEDBACK_KINDS = ("def", "syn", "std")
N_SYN_VALUES = (1, 3, 5, 10)
FUSION_KINDS = ("vector", "rrf")
RERANK_KINDS = ("top1", "llm")

@dataclass(frozen=True)
class PipelineConfig:
    encoder: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"
    feedback_types: tuple[str, ...] = ("def",)
    n_syn: int = 5
    target_kb: str = "KB"                 # user-supplied; std instruction only
    top_k_retrieval: int = 10
    top_k_concept: int = 5
    fusion: str = "vector"
    rerank: str = "top1"
    llm_model: str = "gpt-4o"
    index_kb: str | None = None           # optional metadata filter
```

`__post_init__` raises `ValueError` for: empty encoder, unknown fusion/rerank,
feedback kind outside `FEEDBACK_KINDS`, duplicates, `n_syn not in
N_SYN_VALUES`, `top_k_retrieval < 1`, `top_k_concept not in 1..10`,
empty `target_kb`/`llm_model`, `index_kb` empty string.

## kb.py  (C1 + C2)

```python
def load_concepts_jsonl(path) -> list[Concept]
def concepts_from_dataset(records) -> list[Concept]
def surface_entries(concepts) -> list[tuple[str, str, str | None]]
    # (cui, surface, source_kb); preferred name first, aliases deduped,
    # homonyms rewritten as: f"{surface} ({concept.name})"
def is_homonym_surface(a: str, b: str) -> bool   # optional helper
```

Rules: surface strings are NFC-normalised and whitespace-collapsed before
comparison; CUIs are stripped and NFC-normalised (not whitespace-collapsed);
one entry per distinct surface form per concept; preferred name always indexed
unchanged; an alias equal (after normalisation) to the preferred name is not
duplicated. An alias that occurs in the alias lists of >1 **distinct CUI** is
disambiguated for **every alias occurrence** as
`f"{alias} ({concept.name})"`; the same CUI repeated across source KBs counts
once. Preferred names are never rewritten and are not counted for this rule
(reference_abstract §3 defines the homonym as an alias-set intersection;
architecture §5.1 rewrites aliases). Rewritten forms are deduplicated per
concept so a rewrite cannot create a duplicate index triple.
`concepts_from_dataset` builds a surrogate KB from the gold file: name = most
frequent `vi_term` per CUI (ties resolved by first appearance), aliases = the
other observed terms, `source_kb=None`. Document leakage in its docstring.

## data.py

```python
@dataclass(frozen=True)
class MentionSample:
    mention: str
    context: str
    gold_cui: str
    label: str | None
    start_token: int | None
    end_token: int | None
    record_index: int
    entity_index: int

@dataclass(frozen=True)
class DataIssue:
    record_index: int
    entity_index: int | None
    code: str            # e.g. "bad_record", "bad_entity", "span_mismatch"
    detail: str

@dataclass(frozen=True)
class DatasetLoadResult:
    samples: tuple[MentionSample, ...]
    issues: tuple[DataIssue, ...]

def load_bel_jsonl(path, *, strict: bool = False) -> DatasetLoadResult
```

Mention text comes from `vi_term`. Gold CUI and mention are stripped and
NFC-normalised (the supplied Vietnamese file mixes NFC and NFD). Token indices
are optional metadata and are validated against `context.split()` only when
present, with both observed and expected spans NFC-normalised; a mismatch adds
a `span_mismatch` issue and does **not** change the mention. `strict=True`
raises `DataValidationError` on structural errors (bad record / missing cui /
missing vi_term / empty context); span mismatches remain issues.

## encoder.py  (C3)

```python
class Encoder(Protocol):
    @property
    def dim(self) -> int: ...
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """(n, dim) float32, L2-normalised; empty input -> (0, dim)."""

class SentenceTransformerEncoder:
    def __init__(self, model_name: str, *, batch_size: int = 32,
                 device: str | None = None) -> None
    @property dim
    def encode(self, texts) -> np.ndarray

def build_encoder(config: PipelineConfig) -> SentenceTransformerEncoder
```

Lazy `import sentence_transformers` inside `__init__`; raise `RuntimeError`
with install guidance if absent. Validate non-empty texts, 2-D output, finite
values, non-zero norms.

## index.py  (C4)

```python
class VectorIndex:
    def __init__(self, vectors, cuis, surfaces, source_kbs=None, *,
                 encoder_info=None): ...
    @classmethod
    def from_concepts(cls, concepts, encoder, *, batch_size=64,
                      encoder_info=None) -> "VectorIndex"
    def search(self, query, top_k: int, *, source_kb: str | None = None
               ) -> list[RetrievalHit]
    def save(self, directory) -> None
    def load(self, directory) -> "VectorIndex"     # classmethod
    def __len__(self) -> int
```

Exact cosine scan (architecture permits exact). Deterministic ordering:
score desc, then `cui`, then `surface`. Empty index -> `[]`, but query
dimension is checked before the empty short-circuit. Save/load:
`vectors.npy` + `meta.json` (no pickle); `encoder_info` is optional metadata
(`{"encoder": ..., "model": ...}`) and older files without it load with
`encoder_info=None`. Raise `ValueError` on shape/length mismatch, string passed
where a sequence is expected, non-finite vectors, zero vectors, bad `top_k`.

## fusion.py  (C6)

```python
def component_weights(n_feedback_components: int) -> tuple[float, tuple[float, ...]]
    # returns (w0, w_f...), each 1/(n+1); n=0 -> (1.0, ())
def interpolate_vectors(q0, feedback_vectors, weights) -> np.ndarray
    # Q_GRF = w0*Q0 + sum(w_i*Q_Fi); returns L2-normalised vector
def rrf_scores(rank_lists, weights) -> dict[str, float]
    # rank_lists: sequence of {cui: 1-based rank}; missing -> 0 contribution
```

All weights sum to 1. Validate shapes, finiteness, positive ranks.

## aggregation.py  (C7)

```python
def aggregate_hits(hits, top_k_concept: int | None = None) -> list[Candidate]
def concept_rank_list(hits) -> dict[str, int]
```

`Score(c) = max S(a)` over retrieved aliases of `c`; representative surface is
that of the maximum-scoring hit. Sort score desc, then `cui` asc.
`concept_rank_list` is 1-based over the same ordering.

## feedback.py  (C5) + llm.py

```python
class LLMClient(Protocol):
    def complete(self, prompt: str, *, timeout_s: float | None = None) -> str: ...

class OpenAICompatibleClient:      # stdlib urllib, no openai dependency
    def __init__(self, model: str, *, api_key=None, base_url=None,
                 temperature: float = 0.0, timeout_s: float = 30.0): ...
    def complete(self, prompt, *, timeout_s=None) -> str   # raises LLMError

class FeedbackCache(Protocol):
    def get(self, key) -> tuple[str, ...] | None: ...
    def set(self, key, texts) -> None: ...

class InMemoryFeedbackCache: ...

class FeedbackGenerator:
    def __init__(self, llm: LLMClient, *, cache=None, max_workers: int = 3): ...
    def generate(self, mention, context, feedback_types, n_syn, target_kb
                 ) -> tuple[tuple[Feedback, ...], tuple[FeedbackFailure, ...]]

def build_instruction(kind, mention, context, n_syn, target_kb) -> str
def build_prompt(mention, context, instruction) -> str
def parse_synonyms(text: str, n: int) -> tuple[str, ...]
```

Prompt template exactly:
`Context: {s}\nMention: {m}\nInstruction: {t}`.
Instruction texts are copied verbatim from reference_abstract §4. Feedback
calls run concurrently; cache key = (mention, context, kind, instruction);
only successful results are cached. Any per-kind exception becomes a
`FeedbackFailure`; the generator never raises for LLM failures. Unknown kind /
invalid n_syn raise `ValueError` before calling. The low-level prompt and
generator APIs accept any positive integer `n_syn`; the reference value set
`N_SYN_VALUES` is enforced by `PipelineConfig` and `EntityLinker.link()`.

## rerank.py  (C8)

```python
class Reranker(Protocol):
    def rerank(self, mention: str, context: str,
               candidates: Sequence[Candidate],
               concept_names: Mapping[str, str]) -> str: ...

class PickTopOne: ...
class LLMReranker:
    def __init__(self, llm: LLMClient, *, timeout_s: float = 30.0): ...
```

`PickTopOne` returns the CUI with the largest finite `Candidate.score`, with
`cui` ascending as the tie-break (architecture §8); it does not assume the
sequence is pre-sorted. It raises `ValueError` if empty or a score is invalid.
`LLMReranker` may use only candidates + `concept_names` (name text supplied by
the caller; the schema carries no definitions) + mention/context; it must not
touch the index. It returns a CUI from the candidate list or raises
`RerankError`.

## pipeline.py + evaluator.py + cli.py

```python
class EntityLinker:
    def __init__(self, index, encoder, config, *,
                 feedback_generator=None, reranker=None,
                 concept_names: Mapping[str, str] | None = None): ...
    def link(self, request: Request) -> Response
    def link_batch(self, requests) -> list[Response]

class Metrics: n, recall_at_1, recall_at_5, accuracy
def evaluate(linker, samples) -> Metrics
def evaluate_all(linker, samples) -> tuple[Metrics, dict[str | None, Metrics]]
def evaluate_by_label(linker, samples) -> dict[str | None, Metrics]
```

Pipeline 1: fuse Q0 + Q_F (weights 1/(n+1)) -> one index scan
(`top_k_retrieval`) -> max aggregation -> `top_k_concept` -> rerank.
Pipeline 2: one index scan per query -> concept rank lists -> weighted RRF ->
`top_k_concept` -> rerank. `feedback_types=()` degrades to dense retrieval.
If every feedback call fails, use Q0 only. If `fusion`/`rerank` are invalid
or `rerank="llm"` without a reranker, construction raises `ValueError`. When
`rerank != "llm"`, an injected reranker is deliberately ignored and
`PickTopOne` is used (the config selects the mode). `target_kb` only fills the
`std` instruction; retrieval filtering uses `index_kb` (None = all sources).
`Response.debug` includes `Q_0`, `Q_F`, weights, feedback texts, failures,
retrieval hits, the untruncated concept ranking (`concept_ranking`, used by the
evaluator for R@5), `score_kind`, and rerank mode. `rank_lists` and
`rrf_scores` are present only in the RRF pipeline. The CLI records
`encoder_info` when building an index and rejects a prebuilt index whose
stored encoder differs from the requested one.

CLI: `python -m paper_method.cli {build-index,normalize,evaluate}`. Wiring only.

## No-execution rule

No tests are shipped. Do not execute Python, pytest, linters, installers, or
any script. Static inspection only.
