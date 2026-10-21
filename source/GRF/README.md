# GRF — Biomedical Entity Linking

Implementation of the methods in [`docs/architecture.md`](../../docs/architecture.md)
and [`paper/reference_abstract.md`](../../paper/reference_abstract.md). No
pipeline code in this package was executed during implementation; no empirical
results are claimed. The static audit record is in [`AUDIT.md`](AUDIT.md).

## Component map

| Arch. | Responsibility | Module | Reference |
| --- | --- | --- | --- |
| C1 | KB loader/parser → `Concept(cui, name, aliases)` | `common/kb.py` (`load_concepts_jsonl`, `concepts_from_dataset`) | §2 |
| C2 | Homonym rewriting `a + " (" + N_c + ")"` | `common/kb.py` (`surface_entries`) | §3 |
| C3 | Encoder `E(·)` (BioSyn/SapBERT) | `common/encoder.py` | §3, §5 |
| C4 | Exact cosine vector index, one entry per surface form; directory (`save`/`load`) or single-file `.npz` (`save_npz`/`load_npz`) persistence | `common/index.py` | §3, §4.2, §5.1 |
| C5 | Feedback generation (`def`, `syn`, `std`) + cache + fallback | `feedback.py`, `llm.py` | §4, §6 |
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

- `Q_0 = E(m)`; `Q_F = E(F)`; for synonyms
  `Q_F = (1/n) Σ_j E(F_syn,j)` (mean of the encoder outputs, not renormalised
  before fusion).
- Vector pipeline:
  `Q_GRF = w_0 Q_0 + Σ_i w_i Q_Fi` with every weight `w = 1/(n+1)`, where `n`
  is the number of successful feedback components. The fused vector is
  L2-normalised (cosine ranking is unchanged).
- Rank pipeline: `RRF(c) = w_0 / r_0(c) + Σ_i w_i / r_Fi(c)`, with
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
would need a custom `Encoder` adapter. The LLM client is OpenAI-compatible
(`OPENAI_API_KEY`, `OPENAI_BASE_URL`, default model `gpt-4o`) and uses
`temperature=0.0` for determinism. This temperature is an implementation
choice; the reference does not state one. Results cannot be validated against
the paper's tables with the local data.

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
```

Set `OPENAI_API_KEY`/`OPENAI_BASE_URL` for `def`/`syn`/`std` feedback and the
LLM re-ranker. Without a reachable LLM, feedback failures degrade to `Q_0`
and LLM re-rank failures fall back to top-1, both recorded in
`Response.debug`. Indexes built by the CLI store
`encoder_info={"model": <huggingface id>}`; loading a prebuilt index with a
different `--encoder` raises instead of silently mixing vector spaces.
The CLI prints a warning for older indexes without that metadata. `R@5` uses
the untruncated concept ranking, so `top_k_concept` no longer caps it; keep
`top_k_retrieval` at least 5 when comparing with the paper.

## Notebooks

No tests are shipped. The pipeline has never been executed. Colab notebooks live
in [`source/notebooks/GRF/`](../notebooks/GRF/), except the dense baseline,
which is in [`source/notebooks/baseline/`](../notebooks/baseline/):

- `build_KB.ipynb` embeds the KB JSONL from Google Drive into the single-file
  artifact.
- `retrieval.ipynb` loads the KB data, the artifact and the gold data from
  Drive, runs candidate retrieval, reports R@1/R@5, and saves every mention's
  candidates to one JSONL file (`candidates.save_candidates`).
- `baseline.ipynb` encodes each mention and scans the same artifact directly
  (`common.encoder` + `common.index`), with no feedback, fusion or LLM. It
  reports R@1/R@5/accuracy and writes a candidate file in the same format.
- `e2e.ipynb` reads only that candidate file, applies Top-1 or the LLM
  re-ranker (`candidates.rerank_candidates`), and reports end-to-end accuracy;
  it never loads the encoder, the index or the KB artifact.

Every long stage (KB parsing, embedding, retrieval, candidate save/reload, gold
data loading, re-ranking) drives a `tqdm` bar through the library's
`progress_callback`; all four notebooks install `tqdm` explicitly.

With `DEVICE = "cuda"`, `retrieval.ipynb` and `baseline.ipynb` keep one GPU copy of the index
(`VectorIndex.use_gpu`) and scores queries with a batched matrix multiply; only
rows at or above each k-th score return to CPU. If the copy does not fit, it
logs a warning and uses the CPU scan. Each notebook ends by dropping its large
objects (`gc.collect`); `build_KB`, `retrieval` and `baseline` also empty the
CUDA cache (`release_gpu`, `torch.cuda.empty_cache`), while `e2e` has no GPU
state.
