# Cấu trúc lưu trữ

```text
COMMON_DIR/
├── data/
│   ├── kb.csv
│   ├── translated.jsonl
│   └── {TRANSLATE_NAME}/
│       └── {ENCODER_NAME}/
│           └── grf_kb.npz
├── notebooks/
│   ├── {ENCODER_NAME}/
|   |   └── {TRANSLATE_NAME}/
│   │       ├── baseline/
|   |       │   ├── ranking.ipynb
│   │       │   └── reranking.ipynb
│   │       ├── grf/
│   │       │     ├── ranking.ipynb
│   │       │     └── reranking.ipynb
│   │       ├── {OTHER_METHOD}
│   │       └── buid_kb.ipynb
│   └── common/...
└── output/
    └── {ENCODER_NAME}/
        └── {TRANSLATE_NAME}/
            ├── ranking/
            |    ├── baseline/
            |    │   └── baseline_candidates.jsonl
            |    ├── grf/
            |    │   ├── grf_feedback_bundle.jsonl
            |    │   ├── grf_feedback.jsonl
            |    │   ├── candidates/
            |    │   │   └── grf_<scenario>_n{N_SYN}_k{TOP_K_CONCEPT}_candidates.jsonl
            |    │   └── grf_retrieval_summary.json
            |    └── {OTHER_METHOD}
            └── reranking/
                └── {RERANKER_NAME}/
                    ├── baseline/
                    │   └── baseline_e2e_results.json
                    └── grf/
                        ├── e2e_results/
                        │   └── grf_<scenario>_n{N_SYN}_k{TOP_K_CONCEPT}_e2e.json
                        └── grf_e2e_summary.json
```

```text
{ENCODER_NAME}   = ENCODER_MODEL.split("/")[-1]
{TRANSLATE_NAME} = TRANSLATE_MODEL.split("/")[-1]
{RERANKER_NAME}  = <reranker model>.split("/")[-1]
```

## Cache `.npy` của KB

`grf_kb.npz` là file nén duy nhất trên Drive. Khi `ranking.ipynb` nạp index,
`VectorIndex.load_npz` stream member `vectors.npy` ra đĩa **local** rồi để
`pycauset` memory-map file đó — KB không nằm trọn trong RAM:

```text
{KB_NPY_CACHE_DIR}/                  # mặc định tempfile.gettempdir()/grf_kb_npy
└── grf_kb-{SIZE}-{MTIME_NS}/
    └── vectors.npy                  # float32, unit-norm, mmap
```

Tên thư mục chứa `(kích thước, mtime)` của `.npz`, nên build lại KB sẽ extract
lại thay vì dùng vector cũ. Xoá thư mục cache bất cứ lúc nào — lần chạy sau tự
dựng lại.

