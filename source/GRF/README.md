# GRF — Biomedical Entity Linking

Implementation of the methods in [`docs/architecture.md`](../../docs/architecture.md)
and [`paper/reference_abstract.md`](../../paper/reference_abstract.md). No
pipeline code in this package was executed during implementation; no empirical
results are claimed. The static audit record is in [`AUDIT.md`](AUDIT.md).

## Component map

| Arch. | Responsibility | Module | Reference |
| --- | --- | --- | --- |
| C1 | KB loader/parser → `Concept(cui, name, aliases)` | `common/kb.py` (`load_concepts_jsonl`, `load_concepts_csv`, `concepts_from_dataset`) | §2 |
| C2 | Homonym rewriting `a + " (" + N_c + ")"` | `common/kb.py` (`surface_entries`) | §3 |
| C3 | Encoder `E(·)` (BioSyn/SapBERT) | `common/encoder.py` | §3, §5 |
| C4 | Exact cosine vector index, one entry per surface form; directory (`save`/`load`) or single-file `.npz` (`save_npz`/`load_npz`) persistence | `common/index.py` | §3, §4.2, §5.1 |
| C5 | Feedback generation (`def`, `syn`, `std`) + cache + fallback | `feedback.py`; prompt text in `prompts.yaml` (loaded by `prompts.py`); clients in `llm.py` (API) and `hf_llm.py` (local transformers) | §4, §6 |
| C6 | Vector interpolation / weighted RRF | `fusion.py` | §5, §7 |
| C7 | Concept aggregation `max` over aliases | `common/aggregation.py` | §7.3 |
| C8 | Pick Top-1 / LLM re-ranker (no index access) | `rerank.py` | §8 |
| C9 | R@1, R@5, end-to-end accuracy | `common/evaluator.py` (metrics), `evaluator.py` (runtime) | §10 |
| — | Pipeline 1/2 orchestration | `pipeline.py` | §5.2, §5.3 |
| — | Candidate snapshot (retrieval → E2E hand-off) | `common/candidates.py` (I/O), `candidates.py` (re-rank) | §10 |

`source/common/` has no dependency on `GRF`: it holds the encoder, vector
index, KB/gold loaders, candidate I/O, aggregation and metrics. The GRF modules
of the same name re-export those symbols, so existing `GRF.*` imports keep
working. The dense baseline notebook uses `common/` only.

## Method equations implemented

- `Q_0 = E(m)`; with `use_context_query=True` it is **only**
  `mean-pool(mention token states in its sentence)` (architecture §4.4) — no
  `E(m)` term is added. It is passed to the fusion **raw**. `Q_F = E(F)`; for
  synonyms `Q_F = (1/n) Σ_j E(F_syn,j)` (mean of the raw encoder outputs).
- **A retrieval vector is L2-normalised exactly once, after `Q_GRF`.** The
  components enter the sum unscaled (`encoder.encode_raw`,
  `encode_mention_in_context`) and the single division happens after the
  weighted sum, in `fusion.interpolate_matrix` / `interpolate_vectors`.
  `encoder.encode` (normalised) only builds the KB index rows and a
  stand-alone query; `VectorIndex` leaves rows that arrive unit-length
  untouched, so the fusion's normalisation is never repeated.
- Vector pipeline:
  `Q_GRF = w_0 Q_0 + Σ_i w_i Q_Fi` with every weight `w = 1/(n+1)`, where `n`
  is the number of successful feedback components; the sum is then
  L2-normalised, once.
- Rank pipeline: each mention has `n+1` separate query vectors (`Q_0`, then one
  per feedback component); the batch is laid out **by component**, so `n+1`
  query matrices are scanned in turn — batch 0 holds every mention's `Q_0`,
  batch `i` every mention's `i`-th feedback vector — and each scan returns its
  own rank matrix. Then `RRF(c) = w_0 / r_0(c) + Σ_i w_i / r_Fi(c)`, with
  `w_i = 1/(n+1)` and `1/r_x(c) = 0` when `c` is absent from list `x`.
  Concept rank lists are built with the same max-over-aliases aggregation.
- Concept score: `Score(c) = max_{a ∈ A_c ∩ Top-k} S(a)`.
- `feedback_types=()` degrades both pipelines to plain dense retrieval.
- Metrics: `evaluate_all` produces overall and per-label R@1/R@5/accuracy
  from a single pipeline pass, so failed (uncached) LLM feedback cannot make
  the two reports disagree. R@1/R@5 use the untruncated concept ranking
  recorded in `debug["concept_ranking"]`, not the `top_k_concept`-truncated
  candidate list passed to the re-ranker.
- Homonym rewriting applies to an alias found in the alias lists of more than
  one distinct CUI (architecture §5.1); preferred names are indexed unchanged
  and do not trigger rewriting. Strings and CUIs are NFC-normalised so visually
  identical Vietnamese NFC/NFD forms compare equal.

### Resolved ambiguity: the two meanings of `n`

`reference_abstract.md` uses `n` both for the number of generated synonyms and
for the number of feedback components in the fusion weights. The architecture
(§7, §9) is explicit: one component per feedback type, each with weight
`1/(n+1)`, and `n=1 → α=0.5`. This implementation therefore uses:

- `n` in the fusion weights = number of successful feedback **components**
  (`def`, `syn`, `std`), each synonym group counting once;
- `n_syn` ∈ {1, 3, 5, 10} = number of synonyms requested from the LLM.

Failed LLM calls are excluded from `n`; if every call fails, the pipeline uses
`Q_0` alone, matching the fallback in architecture §6.

## Configuration

`PipelineConfig` defaults: `encoder="cambridgeltl/SapBERT-from-PubMedBERT-fulltext"`,
`feedback_types=("def",)`, `n_syn=5`, `target_kb="KB"` (user-supplied),
`top_k_retrieval=10`, `top_k_concept=5`, `fusion="vector"`, `rerank="top1"`,
`llm_model="gpt-4o"`. `n_syn` must be one of 1/3/5/10; `top_k_concept` must be
1–10. The default `feedback_types` and `rerank` are working defaults, not
values prescribed by the reference (which evaluates several settings).
`target_kb` is used only in the `std` instruction; it does **not** filter
retrieval. Set `index_kb` explicitly to restrict an index that contains
multiple source KBs (`index_kb=None` searches all entries).

`encoder` is any HuggingFace model id, e.g.
`cambridgeltl/SapBERT-from-PubMedBERT-fulltext` or
`dmis-lab/biosyn-sapbert-bc5cdr-disease`. Models are loaded through
`SentenceTransformer`, so the BioSyn checkpoint uses the library's default
pooling rather than BioSyn's own inference script; strict BioSyn reproduction
would need a custom `Encoder` adapter.

Feedback and re-ranking accept either backend, both behind the same
`LLMClient` protocol:

- `OpenAICompatibleClient` (`llm.py`) — HTTP, OpenAI-compatible endpoint
  (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, default model `gpt-4o`), stdlib only,
  `temperature=0.0` for determinism.
- `TransformersChatClient` (`hf_llm.py`) — weights loaded through
  `transformers` from a Hub id or a local directory, no API key and no network
  at inference time. Loaded lazily, kept resident, reusable chat template
  (disable with `use_chat_template=False`), greedy when
  `temperature=0.0`. `device="auto"` picks CUDA when the runtime has it, and
  `dtype="auto"` is resolved from that resolved device (`bfloat16` on CUDA,
  `float32` on CPU) — deliberately **not** the checkpoint's own declared dtype,
  which is often `float32` and would load a 7B model at ~28 GB. Pass
  `"float16"`/`"bfloat16"` explicitly for a half-precision copy regardless of
  device. Setup failures and generation failures (OOM, unsupported dtype) all
  raise `LLMError`, so they degrade exactly like a failed API call. `unload()`
  frees the weights.

The API temperature is an implementation choice; the reference does not state
one. Results cannot be validated against the paper's tables with the local
data.

The LLM re-ranker sees the mention, context, candidate CUIs, preferred names
and surface forms. The `Concept` schema and KB loader have no definition
field, so definitions cannot be supplied even if an external KB contains them
without a schema/loader change (architecture §8 mentions names or definitions);
this is an implementation limitation, not just a property of the local data.

## Data reality and limitations

The repository contains exactly one dataset:
`data/test_fixed_verified_gold_revised.jsonl` (1,476 sentences, 3,686 annotated
mentions, 1,299 CUIs). Each record has `vi_sentence_str` and entities with
`cui`, `icd10`, `label`, `vi_start_token_idx`, `vi_end_token_idx`, `vi_term`.
Labels are `DISEASE`, `SYMPTOM`, `TREATMENT`, `DIAGNOSTIC`, `CAUSE`; 946
entities have `icd10=null`. Token indices are **inclusive** and validated
against `context.split()`; `vi_term` is used as the mention. The file mixes
NFC and NFD Vietnamese spellings (for example `đau nhức` vs `đau nhức`), so
both `data` and `kb` normalise to NFC before comparing. A small number of
annotations are clause-length spans rather than entity names (for example
physical lines 633, 701, 796, 1069, 1073, 1240, 1265); they are kept as gold
samples but are a data-quality limitation. Span problems are reported as
`DataIssue` entries and never drop samples.

There is **no KB file**, no preferred names, no aliases, and no candidate
labels beyond the gold CUI. `kb.concepts_from_dataset` therefore constructs a
surrogate KB: preferred name = most frequent `vi_term` per CUI; aliases = the
other observed terms. This KB is derived from the gold annotations themselves,
so using it to evaluate the same file leaks the mentions and inflates recall.
It is **not** a user-supplied KB, and the 8 datasets
and Tables 1–3 in the reference cannot be reproduced from this data.

No surface form in this file maps to more than one CUI, so C2 is implemented
but never triggered by this data. Constructing an index with a real KB JSONL
(`cui`, `name`, `aliases`) exercises the homonym path.

## Usage (wired, not executed)

The examples use Bash syntax. On PowerShell, set `$env:PYTHONPATH = "source"`
and put each command on one line instead of using `\` continuations.

```bash
pip install -r source/GRF/requirements.txt
export PYTHONPATH=source

# Build an index from a real KB JSONL.
python -m GRF.cli build-index --kb-jsonl kb.jsonl \
  --encoder cambridgeltl/SapBERT-from-PubMedBERT-fulltext --out index/

# Or from the provided gold file (surrogate KB, leakage warning printed).
python -m GRF.cli build-index --data data/test_fixed_verified_gold_revised.jsonl \
  --encoder cambridgeltl/SapBERT-from-PubMedBERT-fulltext --out index-dataset/

# Link one mention (baseline; empty feedback types).
python -m GRF.cli normalize --index index/ --mention "bệnh phong" \
  --context "..." --feedback-types "" --rerank top1

# Evaluate R@1/R@5/accuracy.
python -m GRF.cli evaluate --data data/test_fixed_verified_gold_revised.jsonl \
  --data-as-kb --encoder cambridgeltl/SapBERT-from-PubMedBERT-fulltext \
  --feedback-types "" --limit 10

# Same, with a local HuggingFace model for feedback and re-ranking.
python -m GRF.cli evaluate --data data/test_fixed_verified_gold_revised.jsonl \
  --data-as-kb --encoder cambridgeltl/SapBERT-from-PubMedBERT-fulltext \
  --feedback-types def --rerank llm --llm-backend local \
  --llm-model Qwen/Qwen2.5-7B-Instruct --llm-device cuda --limit 10
```

For the API backend, set `OPENAI_API_KEY`/`OPENAI_BASE_URL` for
`def`/`syn`/`std` feedback and the LLM re-ranker. For the local backend, pass
`--llm-backend local --llm-model <hf-id-or-path>` (plus `--llm-device`,
`--llm-dtype`); nothing else is needed and no key is read. Without a usable
LLM, feedback failures degrade to `Q_0`
and LLM re-rank failures fall back to top-1, both recorded in
`Response.debug`. Indexes built by the CLI store
`encoder_info={"model": <huggingface id>}`; loading a prebuilt index with a
different `--encoder` raises instead of silently mixing vector spaces.
The CLI prints a warning for older indexes without that metadata. `R@5` uses
the untruncated concept ranking, so `top_k_concept` no longer caps it; keep
`top_k_retrieval` at least 5 when comparing with the paper.

## Notebooks

No tests are shipped. The pipeline has never been executed. Colab notebooks live
in [`source/notebooks/`](../notebooks/): the KB builder is in `common/`, the
GRF notebooks are in `GRF/`, and the dense baseline is in `baseline/`.

The GRF run is split at the feedback phase, and the split is a file:
`feedback.save_feedback_outcomes` writes one JSONL record per mention
(`def`/`syn`/`std` texts plus the failures that degraded to `Q_0`), pinned to
`<model>|<prompt digest>`; `load_feedback_outcomes` reads it back as the
`feedback_outcomes` list `link_batch` accepts.

- `common/build_KB.ipynb` embeds the two-column KB CSV (`CUI,concept`) from
  Google Drive into the single-file artifact. A concept named `NaN` (for
  example C0549184) is the source KB's "none/no" answer and is embedded like
  any other row.
- `feedback.ipynb` loads the gold data, runs `run_feedback_batch` — phase 2
  alone, no encoder and no vector index — and writes that bundle. The model's
  weights are released before the notebook ends, so it holds a 32B LLM and
  nothing else.
- `ranking.ipynb` loads the bundle, the KB data, the artifact and the gold
  data from Drive, runs candidate retrieval over the saved feedback, reports
  R@1/R@5, and saves every mention's candidates to one JSONL file
  (`candidates.save_candidates`). It builds no LLM client at all: an
  `EntityLinker` without a `feedback_generator` replays the bundle.
- `baseline/ranking.ipynb` encodes each mention and scans the same artifact
  directly (`common.encoder` + `common.index`), with no feedback, fusion or
  LLM. It reports R@1/R@5/accuracy and writes a candidate file in the same
  format.
- `reranking.ipynb` reads only that candidate file, applies Top-1 or the LLM
  re-ranker (`candidates.rerank_candidates`), and reports end-to-end accuracy;
  it never loads the encoder, the index or the KB artifact.

Every long stage (KB parsing, embedding, retrieval, candidate save/reload, gold
data loading, re-ranking) drives a `tqdm` bar through the library's
`progress_callback`; all five notebooks install `tqdm` explicitly,
`feedback` installs `transformers` for the local LLM backend, and `reranking`
does too (`baseline/ranking` installs neither — it uses no LLM).

With `DEVICE = "cuda"`, `ranking.ipynb` and `baseline/ranking.ipynb` keep one
GPU copy of the index (`VectorIndex.use_gpu`) and scores queries with a batched
matrix multiply; only rows at or above each k-th score return to CPU. If the
copy does not fit, it logs a warning and uses the CPU scan.

**The LLM and the index never share a card.** `feedback.ipynb` generates every
mention's feedback, unloads the model and empties the CUDA cache; it loads no
encoder and no index. `ranking.ipynb` then takes the index's GPU copy and runs
`link_batch(..., feedback_outcomes=...)`, which replays the bundle through the
encoding, retrieval and ranking phases. No single process ever holds the
feedback weights and the vector copy at the same time.

Every generated feedback text is also appended to `FEEDBACK_CACHE_PATH`
(`JsonlFeedbackCache`, on Drive next to the bundle), so an interrupted
`feedback.ipynb` run keeps what it already paid for and a re-run reloads
instead of regenerating. Both the cache and the bundle use the same namespace,
`<model>|<prompt digest>`, so switching model or editing `prompts.yaml`
invalidates the old answers instead of silently reusing them — and
`ranking.ipynb` refuses a bundle written under a different one.

Prompt text is data, not code: `source/GRF/prompts.yaml` holds every template
the feedback generator and the re-ranker send, in English, and `prompts.py`
loads it, validates each placeholder, and exposes `digest()` for the cache
namespace. Editing a prompt is editing that file — no Python change, no re-run
of anything else.

**Batching.** A local model decodes `batch_size` prompts per `generate` call
(`TransformersChatClient.complete_batch`, `LLM_LOCAL_BATCH_SIZE` in the
notebooks; 4–8 on an 80 GB GPU). This matters because the client holds a lock
across each `generate`: many concurrent `complete` calls would only queue, so
one padded batch is worth far more than eight threads. The re-ranker batches the
same way through `LLMReranker.rerank_batch`.

**GPU residents.** `reranking.ipynb` holds only the re-ranker's model, but that
model is still GPU state when `LLM_BACKEND = "local"` and
`LLM_LOCAL_DEVICE = "cuda"`.

Each notebook ends by dropping its large objects (`gc.collect`) and calling
`unload()` on a local model; `build_KB`, `feedback`, `ranking` and the
baseline's `ranking` also empty the CUDA cache (`release_gpu`,
`torch.cuda.empty_cache`), and `reranking` does the same when a GPU is present.
