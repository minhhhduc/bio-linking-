# Implementation contract — GRF

Frozen interfaces for parallel implementation. Derived from
`docs/architecture.md` and `paper/reference_abstract.md`. Do not change a
signature here without coordinating; other modules import it.

Package roots: `source/common/` (shared encoder, index, KB/gold loaders,
candidate I/O, aggregation, metrics) and `source/GRF/` (feedback, fusion,
re-ranking, orchestration). The GRF modules re-export the common symbols under
their old paths. Python 3.10+. `numpy` is the only hard dependency;
`sentence-transformers` is imported lazily by the encoder.

Progress convention: every record-scanning function the notebooks run takes an
optional `progress_callback(done, total)`; `total` may be `None` when unknown.
Callers that already counted the input pass an explicit `total` so the bar has
a percentage. The notebooks render these callbacks with `tqdm`.
`EntityLinker.link_batch` is the exception: it runs several phases with
different units, so its callback is `progress_callback(phase, done, total)`
(see the pipeline section below).
`concepts_from_dataset` (CLI-only gold-to-KB derivation) has no callback.

## common/schemas.py

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

## common/kb.py  (C1 + C2)

```python
def load_concepts_jsonl(path, *, progress_callback=None,
                        total: int | None = None) -> list[Concept]
def load_concepts_csv(path, *, progress_callback=None,
                      total: int | None = None) -> list[Concept]
def concepts_from_dataset(records) -> list[Concept]
def surface_entries(concepts, *, progress_callback=None
                    ) -> list[tuple[str, str, str | None]]
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

`load_concepts_csv` reads the user-supplied two-column `CUI,concept` CSV
(header case-insensitive): the first concept per CUI is the preferred name,
later distinct concepts become aliases, and a repeated CUI is folded instead of
raising. `NaN` is real KB content (the "none/no" concept, e.g. C0549184), not a
missing value; only an empty field is rejected.

## common/data.py

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

def load_bel_jsonl(path, *, strict: bool = False, progress_callback=None,
                   total: int | None = None) -> DatasetLoadResult
```

Mention text comes from `vi_term`. Gold CUI and mention are stripped and
NFC-normalised (the supplied Vietnamese file mixes NFC and NFD). Token indices
are optional metadata and are validated against `context.split()` only when
present, with both observed and expected spans NFC-normalised; a mismatch adds
a `span_mismatch` issue and does **not** change the mention. `strict=True`
raises `DataValidationError` on structural errors (bad record / missing cui /
missing vi_term / empty context); span mismatches remain issues.

## common/encoder.py  (C3)

```python
class Encoder(Protocol):
    @property
    def dim(self) -> int: ...
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """(n, dim) float32, L2-normalised; empty input -> (0, dim)."""
    def encode_raw(self, texts: Sequence[str]) -> np.ndarray:
        """The same rows before that normalisation (E(·) as the model emits it)."""

class SentenceTransformerEncoder:
    def __init__(self, model_name: str, *, batch_size: int = 32,
                 device: str | None = None) -> None
    @property dim
    def encode(self, texts) -> np.ndarray       # = encode_raw, then L2-normalise
    def encode_raw(self, texts) -> np.ndarray   # model output, untouched
    def encode_mention_in_context(self, mentions, contexts, spans) -> np.ndarray
        # Q_0 = mean-pool(mention token states in the sentence); raw, not normalised

def build_encoder(config: PipelineConfig) -> SentenceTransformerEncoder
```

Lazy `import sentence_transformers` inside `__init__`; raise `RuntimeError`
with install guidance if absent. Validate non-empty texts, 2-D output, finite
values, non-zero norms.

**A retrieval vector is L2-normalised exactly once — after `Q_GRF`.** The
components of the fusion (`Q_0`, every `Q_F_i`) are raw: `encode_raw` hands back
`E(·)` and `encode_mention_in_context` hands back the mean-pooled mention tokens,
both unscaled, and `fusion.interpolate_matrix` divides once, after the weighted
sum. Normalising a component first would scale it *and* hand it a fixed 1:1
weight against the others. `encode` (normalised) is for what is compared rather
than summed: the KB index rows, and a stand-alone query. `VectorIndex` likewise
leaves rows that arrive unit-length untouched.

## common/index.py  (C4)

```python
class VectorIndex:
    def __init__(self, vectors, cuis, surfaces, source_kbs=None, *,
                 encoder_info=None): ...
    @classmethod
    def from_concepts(cls, concepts, encoder, *, batch_size=64,
                      encoder_info=None, progress_callback=None,
                      entries=None) -> "VectorIndex"
        # entries: reuse a caller-computed surface_entries() expansion
    def search(self, query, top_k: int, *, source_kb: str | None = None
               ) -> list[RetrievalHit]
    def search_batch(self, queries, top_k: int, *,
                     source_kb: str | None = None
                     ) -> list[list[RetrievalHit]]
    def use_gpu(self, device: str = "cuda") -> None   # lazy torch import
    def release_gpu(self) -> None                     # drop device copy
    @property
    def search_device(self) -> str                    # "cuda" | "cpu"
    def save(self, directory) -> None
    def load(self, directory) -> "VectorIndex"     # classmethod
    def save_npz(self, path) -> None                # one .npz, no pickle
    def load_npz(self, path) -> "VectorIndex"       # classmethod
    def __len__(self) -> int
```

Exact cosine scan (architecture permits exact). Deterministic ordering:
score desc, then `cui`, then `surface`. Empty index -> `[]`, but query
dimension is checked before the empty short-circuit. `search_batch` scores all
rows with one matrix multiply `Q @ A^T`, chunked on the GPU to bound the
transient `(N, queries)` score matrix; the chunk size shrinks automatically on
large indexes (256 MiB score budget). With `use_gpu()` the vectors live on CUDA,
`topk` runs per query column on the device, and only those k hits cross back to
CPU for the same score/cui/surface tie-break. The ordering rule matches the CPU
path; scores can
differ in the last float32 ULPs because cuBLAS and BLAS reduce differently.
PyTorch is imported lazily and stays optional; `release_gpu()` clears the device
copy and cached CUDA memory. Save/load:
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

## common/aggregation.py  (C7)

```python
def aggregate_hits(hits, top_k_concept: int | None = None) -> list[Candidate]
def concept_rank_list(hits) -> dict[str, int]
```

`Score(c) = max S(a)` over retrieved aliases of `c`; representative surface is
that of the maximum-scoring hit. Sort score desc, then `cui` asc.
`concept_rank_list` is 1-based over the same ordering.

## prompts.yaml + prompts.py  (prompt text is data, not code)

Every prompt the system sends lives in `source/GRF/prompts.yaml` — nothing is
hard-coded, so a prompt change is a file edit, not a code change.

```python
PROMPT_FILE = Path(__file__).with_name("prompts.yaml")
def load(path: Path = PROMPT_FILE) -> dict
def digest(path: Path = PROMPT_FILE) -> str          # sha256, first 16 hex chars
def get_prompts(path: Path = PROMPT_FILE) -> dict    # cached load
```

Keys: `feedback.envelope` (`{context}` `{mention}` `{instruction}`),
`feedback.instructions.{def,syn,std}` (`{m}`, `{s}`, `{n}`, `{KB}`),
`rerank.candidate_row` (`{index}` `{cui}` `{name}` `{surface}`),
`rerank.instruction` (`{mention}` `{context}` `{candidates}`). `load` validates
every placeholder with `string.Formatter`, so a typo fails at load time instead
of silently sending a broken prompt. `digest()` is what namespaces the feedback
cache: editing the YAML invalidates the cached answers.

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
class JsonlFeedbackCache:
    def __init__(self, path, namespace: str, *, fsync: bool = False): ...
    def __len__(self) -> int                            # entries in this namespace

class FeedbackGenerator:
    def __init__(self, llm: LLMClient, *, cache=None, max_workers: int = 3): ...
    def generate(self, mention, context, feedback_types, n_syn, target_kb
                 ) -> tuple[tuple[Feedback, ...], tuple[FeedbackFailure, ...]]
    def generate_batch(self, jobs, target_kb
                       ) -> list[tuple[tuple[Feedback, ...], tuple[FeedbackFailure, ...]]]

def build_instruction(kind, mention, context, n_syn, target_kb) -> str
def build_prompt(mention, context, instruction) -> str
def parse_synonyms(text: str, n: int) -> tuple[str, ...]

FEEDBACK_BUNDLE_FORMAT = "grf-feedback-v1"

def feedback_namespace(model: str) -> str        # "<model>|<prompt digest>"
def run_feedback_batch(generator, requests, target_kb, *, progress_callback=None
                       ) -> list[tuple[tuple[Feedback, ...], tuple[FeedbackFailure, ...]]]
def save_feedback_outcomes(path, requests, outcomes, *, namespace: str,
                           progress_callback=None) -> None
def load_feedback_outcomes(path, *, namespace: str | None = None,
                           progress_callback=None
                           ) -> tuple[list[Request], list[tuple[tuple[Feedback, ...], tuple[FeedbackFailure, ...]]]]
```

`generate_batch` takes `(mention, context, feedback_types, n_syn)` jobs and
returns one outcome pair per job, in order. It is the fast path: every prompt of
the whole batch is collected, the cache answers what it can, and the misses go
to the client in **one** call (`complete_batch` when the client has it, else a
thread pool of `max_workers`). `generate` is a one-job wrapper over it. A
batch-level failure degrades every pending item instead of raising; a per-kind
failure is still a `FeedbackFailure` on that item alone.

`run_feedback_batch` is that phase on its own for a whole request list, with no
encoder, index or KB in reach — the LLM half of the GRF run, so it can live in a
notebook by itself. `save_feedback_outcomes` / `load_feedback_outcomes` are the
file handoff between the two notebooks: one JSONL record per mention
(`format`, `ns`, `mention`, `context`, `start_token`, `end_token`,
`feedback: [{kind, texts}]`, `failures: [{kind, reason}]`), written in request
order and pinned to `feedback_namespace(model)`. The loader rebuilds the
requests with `feedback_types=()`, because the feedback is already in the file
and retrieval must not need a generator; a record from another namespace raises
instead of being replayed.

`JsonlFeedbackCache` is the on-disk cache: append-only JSONL, one object per
line (`ns`, `mention`, `context`, `kind`, `texts`), flushed per write so an
interrupted run keeps what it paid for. `namespace` is
`"<model>|<prompt digest>"`; lines written under another namespace are ignored
on load and left on disk, so switching model or editing `prompts.yaml`
regenerates instead of silently reusing an answer the new prompt would not have
given. A malformed trailing line (killed mid-write) is skipped, and the last
line for a key wins.

`hf_llm.py` is the second `LLMClient` implementation, so `FeedbackGenerator`
and `LLMReranker` are backend-agnostic:

```python
DTYPES = ("auto", "float16", "bfloat16", "float32")   # imported by the CLI

class TransformersChatClient:      # lazy torch/transformers import
    def __init__(self, model_name, *, device="auto", dtype="auto",
                 max_new_tokens=256, temperature=0.0,
                 trust_remote_code=False, use_chat_template=True,
                 batch_size: int = 4): ...
    def complete(self, prompt, *, timeout_s=None) -> str
    def complete_batch(self, prompts) -> list[str]
    def unload(self) -> None
    @property
    def loaded_device(self) -> str | None
```

`complete_batch` decodes `batch_size` prompts per `generate` call: decoding is
memory-bandwidth bound, so one padded batch uses the GPU far better than a
sequence of single-row calls, and the per-call overhead is paid once per batch.
Left padding is required with a decoder-only model — the tokens that continue
the sequence must sit flush against the right edge — so `padding_side` is
switched for the tokenisation and restored afterwards, and the answers are
decoded from `input_length` onwards so padding never reaches the output. An
individual empty answer comes back as `""` (that item's failure only);
batch-level failures still raise `LLMError`.

`model_name` is a HuggingFace id or a local directory. Weights load on the
first `complete()` and stay resident until `unload()`; generation is serialised
with a lock because one resident model serves one generation at a time — which
is exactly why callers prefer `complete_batch` over many concurrent `complete`
calls, which would only queue on that lock. `temperature=0.0` is greedy,
matching the API client's default. `device="auto"` resolves to CUDA when the
runtime has one; `dtype="auto"` is then resolved *from that resolved device*
(`bfloat16` on CUDA, `float32` on CPU), *not* the checkpoint's own dtype — the
device is resolved before the weights load, so the default `"auto"`/`"auto"`
pair really does load bfloat16 on a GPU.
`timeout_s` is accepted and ignored: a local generation
cannot be interrupted mid-step without leaving the CUDA context undefined.
`use_chat_template=False` feeds the raw prompt instead of
`tokenizer.apply_chat_template`. The chat template is rendered with
`enable_thinking=False` when it accepts that variable (Qwen3-class checkpoints
reason by default and would spend the whole `max_new_tokens` budget inside a
` thinking` block), with a fallback for templates that reject the kwarg; any
`think` block that still appears is stripped from the decoded answer. Setup failures (missing dependency, device the
runtime does not have, unloadable checkpoint, move-to-device) and generation
failures (OOM, unsupported dtype) all surface as `LLMError`, so they degrade
exactly like a failed API call. The class is not a GPU-free operation: a local
model is a CUDA resident, which matters on a Colab-sized device — the GRF run is
therefore split into two notebooks, and `feedback.run_feedback_batch` (what
`EntityLinker.feedback_batch` delegates to) exists so the model is freed before
the encoder and the index's GPU copy load.

Prompt template exactly: `feedback.envelope` rendered with the context, the
mention and the instruction. Instruction texts are English, live in
`prompts.yaml`, and are a user-editable rendition of reference_abstract §4.
Cache key = (mention, context, kind, instruction); only successful results are
cached. Any per-kind exception becomes a `FeedbackFailure`; the generator never
raises for LLM failures. Unknown kind / invalid n_syn raise `ValueError` before
calling. The low-level prompt and generator APIs accept any positive integer
`n_syn`; the reference value set `N_SYN_VALUES` is enforced by `PipelineConfig`
and `EntityLinker.link()`.

## rerank.py  (C8)

```python
class Reranker(Protocol):
    def rerank(self, mention: str, context: str,
               candidates: Sequence[Candidate],
               concept_names: Mapping[str, str]) -> str: ...

class PickTopOne: ...
class LLMReranker:
    def __init__(self, llm: LLMClient, *, timeout_s: float | None = 30.0): ...
    def rerank_batch(self, items) -> list[str | Exception]
```

`rerank_batch` takes `(mention, context, candidates, concept_names)` items and
returns one entry per item: the chosen CUI, or the exception that item raised.
Every prompt is built first, then decoded in one batched call; each answer is
parsed separately, so one unusable answer degrades that mention only. A
batch-level failure raises `RerankError` and the caller falls back to the
per-mention path. `rerank` remains the single-item API, and `_parse_answer` is
shared by both.

`timeout_s=None` forwards no timeout of its own, which means "use the client's
policy": the OpenAI client applies its configured default, a local model has
none. `LLMReranker` still validates a numeric timeout when one is given.

`PickTopOne` returns the CUI with the largest finite `Candidate.score`, with
`cui` ascending as the tie-break (architecture §8); it does not assume the
sequence is pre-sorted. It raises `ValueError` if empty or a score is invalid.
`LLMReranker` may use only candidates + `concept_names` (name text supplied by
the caller; the schema carries no definitions) + mention/context; it must not
touch the index. It returns a CUI from the candidate list or raises
`RerankError`.

```python
def select_with_fallback(reranker, mention, context, candidates,
                         concept_names=None) -> tuple[str, str | None]
    # (cui, error); RerankError -> first candidate's CUI, error text.
```

Shared by `EntityLinker.link` and `candidates.rerank_candidates` so the live and
saved-candidate paths degrade identically.

## common/candidates.py + candidates.py (retrieval → E2E hand-off)

`FORMAT`, `save_candidates` and `load_candidates` live in
`common/candidates.py`; `GRF/candidates.py` re-exports them and adds
`rerank_candidates` (which needs the GRF re-ranker).

```python
FORMAT = "grf-candidates-v1"
def save_candidates(path, samples, responses, *, progress_callback=None) -> None
def load_candidates(path, *, progress_callback=None, total: int | None = None
                    ) -> tuple[list[MentionSample], list[Response]]
def rerank_candidates(samples, responses, *, reranker, concept_names=None,
                      progress_callback=None) -> list[Response]
```

One JSONL record per mention: format tag, sample fields (mention, context,
gold CUI, label, token spans, record/entity indices), retrieval `cui`,
`candidates`, the untruncated `concept_ranking`, `score_kind` and `error`.
`load_candidates` validates the sample and candidate fields (the optional
`score_kind`/`error` values pass through) and reconstructs
`Response.debug["concept_ranking"]`, so `metrics_from_responses` can compute
R@1/R@5 without re-running retrieval. `rerank_candidates` applies the re-ranker
and reports progress; it never loads the encoder or the vector index. When the
re-ranker offers `rerank_batch` (an `LLMReranker`), the whole candidate list is
turned into prompts, decoded in batches, and the answers are then assigned back
to their mentions — one invalid answer degrades that mention to Top-1, and a
batch-level failure falls back to the per-mention `select_with_fallback` path.
`save_candidates`/`load_candidates` accept the same `progress_callback`; pass
`total` when the caller already counted the records.

## pipeline.py + evaluator.py + cli.py

Metric computation (`Metrics`, `metrics_from_responses`) lives in
`common/evaluator.py`; `GRF/evaluator.py` re-exports it and adds the
pipeline-running wrappers (`make_requests`, `evaluate`, `evaluate_all`).

```python
class EntityLinker:
    def __init__(self, index, encoder, config, *,
                 feedback_generator=None, reranker=None,
                 concept_names: Mapping[str, str] | None = None,
                 use_context_query: bool = True): ...
    def link(self, request: Request) -> Response
    def feedback_batch(self, requests, *, progress_callback=None
                       ) -> list[tuple[tuple[Feedback, ...], tuple[FeedbackFailure, ...]]]
    def link_batch(self, requests, *, feedback_outcomes=None,
                   progress_callback=None) -> list[Response]
class Metrics: n, recall_at_1, recall_at_5, accuracy
def evaluate(linker, samples) -> Metrics
def evaluate_all(linker, samples, *, progress_callback=None
                 ) -> tuple[Metrics, dict[str | None, Metrics]]
def metrics_from_responses(samples, responses
                           ) -> tuple[Metrics, dict[str | None, Metrics]]
def evaluate_by_label(linker, samples) -> dict[str | None, Metrics]
```

`feedback_batch` runs phase 2 alone and returns exactly what `link_batch`
accepts as `feedback_outcomes`; it delegates to
`feedback.run_feedback_batch`, which is the same phase without a linker. The GRF
notebooks use the split directly: `feedback.ipynb` generates every mention's
feedback and unloads the LLM, `ranking.ipynb` loads the encoder and the index's
GPU copy and replays the bundle — the feedback phase is the only one that needs
no encoder at all, and on a large feedback model the two sets of weights do not
fit side by side. `link_batch(..., feedback_outcomes=...)` replays that list and
skips phase 2, so the `"feedback"` phase is not reported a second time; the list
length must match the request count.

`link_batch` reports through `progress_callback(phase, done, total)` — a
different unit per phase, so a caller can draw `PROGRESS_PHASES` as one bar
each: `"encode Q0"` and `"encode feedback"` count encoder calls with
`total=None` (the encoder batches internally); `"feedback"`, `"aggregation"`
and `"ranking"` count mentions; `"retrieval"` counts index scans — `1` for
vector fusion, `n+1` for RRF, because RRF scans once per query component
(§7.3: `Q_0`, then each `Q_Fi`).

`metrics_from_responses` computes the same overall/per-label metrics from
already-computed responses; `evaluate_all` delegates to it after one
`link_batch` pass, and the retrieval/E2E notebooks use it directly.

Pipeline 1: fuse Q0 + Q_F (weights 1/(n+1)) -> one index scan
(`top_k_retrieval`) -> max aggregation -> `top_k_concept` -> rerank.
Pipeline 2: `n+1` index scans, **one per query component, in order** — batch 0
holds every mention's `Q_0`, batch `i` holds every mention's `i`-th feedback
vector — each returning its own rank matrix; then concept rank lists ->
weighted RRF -> `top_k_concept` -> rerank. `feedback_types=()` degrades to
dense retrieval.
If every feedback call fails, use Q0 only. If `fusion`/`rerank` are invalid
or `rerank="llm"` without a reranker, construction raises `ValueError`. When
`rerank != "llm"`, an injected reranker is deliberately ignored and
`PickTopOne` is used (the config selects the mode). `target_kb` only fills the
`std` instruction; retrieval filtering uses `index_kb` (None = all sources).
`Response.debug` includes `Q_0`, `Q_F`, weights, feedback texts, failures,
retrieval hits, the untruncated concept ranking (`concept_ranking`, used by the
evaluator for R@5), `score_kind`, and rerank mode. The recorded `Q_0` and `Q_F`
are the **raw** vectors that entered the fusion (not unit-normalised), so a
check can recompute `Q_GRF` from the debug entry alone. `rank_lists` and
`rrf_scores` are present only in the RRF pipeline. The CLI records
`encoder_info` when building an index and rejects a prebuilt index whose
stored encoder differs from the requested one.

CLI: `python -m GRF.cli {build-index,normalize,evaluate}`. Wiring only.

## No-execution rule

No tests are shipped. The pipeline, encoder, notebooks and CLI were never
executed; do not run them during static inspection.
