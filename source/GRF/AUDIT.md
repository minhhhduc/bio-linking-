# Static audit record — GRF

Status: static audit and independent read-only re-audits completed on the
current tree. **No project code was executed** at any stage: no Python, tests,
imports, builds, notebooks, linters, formatters, type checkers, dependency
installs, or API calls. File inspection was read-only.

The `tests/` directory mentioned below is not present in the current tree;
those mentions describe the tree at audit time.

## Sources used

- `docs/architecture.md`: C1–C9, data contracts §4, pipelines §5, prompts §6,
  fusion §7, re-ranking §8, parameters §9, metrics §10, degradation §11.
- `paper/reference_abstract.md`: problem statement, homonym rule §3, prompts
  §4, pipelines §5, non-reproducible empirical tables §6.
- `data/test_fixed_verified_gold_revised.jsonl`: 1,476 records, 3,686 mentions,
  1,299 CUIs; mixed NFC/NFD Vietnamese; inclusive token indices; null ICD-10.
- `source/GRF/*.py`, `CONTRACT.md`, `README.md`.

## Independent review and separation

Implementation and audit were separate. The implementation subagents did not
audit their own work. Independent, read-only re-auditors reviewed the current
tree without editing files:

- Method-correctness re-auditor `af1eef8b30159693a`: prompt/instruction text,
  homonym rule, synonym parsing, both pipelines, aggregation, R@1/R@5,
  fallback semantics.
- Edge-case/maintainability re-auditor `a970c0bdc9979dfad`: validation,
  persistence, Unicode, CLI wiring, error paths, test coverage.
- A separate integration/data auditor `a0e56472123e21a1d` was dispatched. It
  produced no report before this record was written; **no findings from it are
  claimed here**. That is an outstanding verification gap, not a clean result.

Both completed auditors explicitly reported that they executed no code. Their
reports were used as evidence, not as runtime proof.

## Confirmed defects found and fixed

| Severity | Location | Defect | Fix / evidence |
| --- | --- | --- | --- |
| High | `evaluator.py:26` | Syntax corruption `accuracy: float///` blocked package import and every test. | Changed to `accuracy: float`; both independent re-audits confirmed the syntax fix. |
| High | `evaluator.py:58-88`, `pipeline.py:218-222` | R@1/R@5 used `Response.candidates`, which is truncated to `top_k_concept`; R@5 was capped. | Evaluator now reads the untruncated `debug["concept_ranking"]`, with nested fallback; both pipelines record the full ranking. |
| High | `evaluator.py:116-125` | A linker returning fewer responses than requests was silently truncated by `zip`, producing incorrect metrics. | `_link` raises when the response count differs. |
| High | `pipeline.py:146-166` | A feedback generator could return the wrong kind, a duplicate, or a missing outcome and silently change fusion. | Exactly one outcome per requested kind is enforced, with tests. |
| Medium | `pipeline.py:230-248` | A non-string or non-candidate re-ranker result could escape validation. | Non-string/non-candidate results raise `RerankError` and fall back to the first candidate. |
| Medium | `fusion.py:27-54` | `(True, 0.0)` bypassed the weight dtype check because NumPy promoted it to `float64`. | Original elements are checked for `bool`/`np.bool_` before casting. |
| Medium | `index.py` save/load, `cli.py:77-92` | A prebuilt index did not record its encoder, so a mismatched checkpoint could silently mix vector spaces. | CLI-built indexes store `encoder_info`; loading checks encoder/model and rejects mismatches. Old indexes warn. |
| Medium | `index.py:89-92` | Duplicate `(cui, surface, source_kb)` triples wasted Top-k slots. | Duplicate triples now raise. |
| Medium | `kb.py:99-106` | Duplicate `(cui, source_kb)` concept records could create inconsistent aliases. | Duplicate concept records now raise. |
| Medium | `cli.py:223-224` | Negative `--limit` sliced samples from the end. | Negative limits now raise. |
| Low | `data.py:279-281`, `kb.py` loaders | Gold CUIs and KB CUIs were not consistently NFC-normalised, so visually identical NFC/NFD forms could fail equality. | CUIs are stripped and NFC-normalised in both paths; tests added. |
| Low | `feedback.py:106-108` | An NFD Vietnamese preamble did not match the NFC preamble regex. | The whole answer is NFC-normalised before preamble removal; NFD synonyms and preambles are tested. |
| Low | `evaluator.py:66-88` | Empty or malformed top-level rankings could produce empty/partial metrics. | The ranking is validated; invalid entries fall back to candidates. Tests cover empty/malformed lists. |
| Low | `evaluator.py:140-170`, `cli.py:260` | Overall and per-label evaluation ran the pipeline twice, so uncached LLM failures could make reports disagree. | `evaluate_all` computes both from one `link_batch` pass. |
| Low | `rerank.py:70-78` | `timeout_s=inf`/`nan` passed validation. | Timeout must be positive and finite. |
| Low | `tests/test_*.py` | Tests depended on the caller's `PYTHONPATH`. | Each test file inserts `source/` into `sys.path`. |

## Independent re-audit outcome

The method re-auditor confirmed that the R@5 truncation, gold-CUI
normalisation, and malformed-ranking findings no longer hold, and found no
regression in the changed pipeline/evaluator/KB/data/feedback paths. The edge
re-auditor confirmed the evaluator syntax and fusion boolean fixes and left no
High-severity correctness finding. Surviving items were documentation
staleness, the risks listed below, and test-coverage gaps; this file replaces
the stale claims.

## Potential risks and unresolved items

- **Partial feedback failure semantics:** with one of several feedback calls
  failing, weights are recomputed over successful components at
  `1/(n+1)` (`pipeline.py:190-203`). Ranking is equivalent to a fixed-weight
  scheme with the failed component contributing zero, but the debug weights
  differ and the sources do not state partial-failure behavior explicitly.
  All-failure degradation to `Q_0` is implemented and tested.
- **Zero-norm fused vector:** `interpolate_vectors` raises if the weighted sum
  has zero norm (`fusion.py:90-92`). Encoder outputs are individually non-zero,
  but cancellation is theoretically possible; this raises instead of
  degrading to `Q_0`.
- **Duplicate surface entries across source KBs:** uniqueness is enforced on
  `(cui, surface, source_kb)`, so the same CUI/surface can appear once per
  source KB. With `index_kb=None`, duplicates can occupy multiple Top-k slots.
  This is intentional for source filtering but affects unfiltered indexes.
- **R@5 observability:** R@5 is no longer capped by `top_k_concept`, but
  `top_k_retrieval < 5` means fewer than five concepts can be retrieved.
  Duplicate aliases of one concept in Top-k can also crowd out distinct
  concepts. Keep `top_k_retrieval >= 5` when comparing with the paper.
- **`target_kb` vs `index_kb`:** `target_kb` only fills the `std` instruction;
  it does not filter retrieval. `index_kb` controls the optional source filter
  (`config.py`, `pipeline.py`). A multi-KB index is not automatically filtered
  to `target_kb`; set `index_kb` explicitly.
- **Encoder metadata gap for old indexes:** indexes without `encoder_info`
  load with a warning, not an error. A mismatched old index can still be used.
- **BioSyn adapter:** both checkpoints load through `SentenceTransformer`;
  BioSyn therefore uses the library's default pooling, not BioSyn's own
  inference path. Strict reproduction needs a custom `Encoder` adapter.
- **No definition text for re-ranking:** `Concept` has no definition field and
  the KB loaders cannot supply one. The LLM re-ranker sees names and surfaces
  only, although architecture §8 permits names or definitions.
- **Direct feedback API `n_syn`:** `FeedbackGenerator`/`build_instruction`
  accept any positive integer; only `PipelineConfig` and
  `EntityLinker.link()` enforce `N_SYN_VALUES`. This is documented in
  `CONTRACT.md` and tested, but callers bypassing the pipeline can request
  values outside the reference set.
- **Injected reranker precedence:** when `config.rerank != "llm"`, an injected
  reranker is ignored and `PickTopOne` is used. This is deliberate and
  documented in `CONTRACT.md`, but can surprise callers.
- **`concept_names` fallback:** `cli._concept_names_from_index` takes the first
  indexed surface per CUI, which may be an alias or a rewritten homonym rather
  than the preferred name. The re-ranker then sees a non-preferred name.
- **Malformed ranking fallback:** if both the top-level and nested concept
  rankings are absent or malformed, `_retrieval_cuis` silently falls back to
  `Response.candidates`. Current `EntityLinker` writes valid copies, so this
  only affects custom linkers; there is no diagnostic for the fallback.
- **Per-label key collision:** the CLI serialises labels with `str(label)`, so
  `None` and the literal string `"None"` would share a JSON key. Current data
  has string labels only.
- **Data issues lack physical line numbers:** `DataIssue` carries record and
  entity indices, not the physical JSONL line; CLI diagnostics point to the
  record rather than the source line.
- **Surrogate-KB leakage and data limits:** the repository has one Vietnamese
  gold file and no real KB, preferred names, aliases, or definitions.
  `concepts_from_dataset` is derived from the gold annotations and leaks
  mentions. No surface in this file maps to more than one CUI, so the homonym
  path is only exercised by an external KB. The paper's 8 datasets and Tables
  1–3 cannot be reproduced from the local data.
- **Scale and cache bounds:** retrieval is an exact cosine scan; the feedback
  cache is an unbounded in-process dictionary. Both are acceptable for the
  reference-scale experiments but are not a production retrieval design.
- **No runtime validation:** behavior is reasoned statically. No tests are
  shipped; no empirical or table results are claimed.
- **Outstanding integration/data audit:** the dedicated integration/data
  auditor produced no report. Integration and data correctness beyond the
  checks listed in this record remain unverified by that independent pass.

## Requirements checked without finding a confirmed defect

- Homonym rewriting `a + " (" + N_c + ")"` for every alias occurrence, using
  distinct-CUI alias intersections; preferred names unchanged; rewritten
  forms deduplicated; NFC-normalised comparisons.
- Verbatim prompt template and Vietnamese `def`/`syn`/`std` instructions.
- Synonym feedback as the mean of encoded synonym vectors.
- Pipeline 1: `Q_0` + feedback interpolation with `1/(n+1)` weights and
  exactly one index scan; max-over-aliases aggregation; `top_k_concept`.
- Pipeline 2: one scan per query, 1-based concept rank lists, weighted RRF
  `1/r`, missing concept contributes zero; `top_k_concept`.
- `feedback_types=()` degrades to dense retrieval; all-failed feedback degrades
  to `Q_0`; LLM re-ranker failure falls back to Top-1.
- Pick Top-1 uses the largest finite score with CUI-ascending tie-break; the
  LLM re-ranker receives candidates and names but no index.
- Inclusive token spans are validated against `context.split()`; null ICD-10
  is ignored without inventing labels.
- Encoder metadata is persisted and checked by the CLI.

## Complexity audit (ponytail, read-only)

Findings only; no cuts applied.

- `yagni: FeedbackCache Protocol has one implementation and is not exported; accept the concrete cache or a duck-typed object. [feedback.py:161-164]`
- `yagni: LLMClient Protocol has one production implementation; type against the concrete client or a callable. [llm.py:25-28]`
- `yagni: Encoder Protocol has one production implementation; keep only if a second encoder adapter is planned. [encoder.py:21-29]`
- `delete: is_homonym_surface has no production caller; its test can exercise the normalisation through surface_entries. [kb.py:27-29]`
- `shrink: evaluate_by_label is a one-line wrapper; call evaluate_all(...)[1] if it is not retained as public API. [evaluator.py:166-170]`

`net: -30 lines, -0 deps possible.` These are simplification candidates, not
correctness defects, and were not applied.

## No-execution confirmation

No application or package code was run. No Python interpreter, test runner,
build, notebook, linter, formatter, type checker, dependency installer, or API
client was invoked. No execution results or empirical performance claims are
made.
