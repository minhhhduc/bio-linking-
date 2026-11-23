# Kiến trúc hệ thống — GRF cho Biomedical Entity Linking

Tài liệu này là **mục lục** của kiến trúc triển khai (implementation architecture) cho phương pháp trong [`paper/reference_abstract.md`](../../paper/reference_abstract.md): dùng Generative Relevance Feedback (GRF) từ LLM để mở rộng truy vấn, nâng trần recall của candidate retrieval và cải thiện xếp hạng cuối.

## 1. Mục tiêu kiến trúc

- Chuẩn hóa mention $m$ trong câu ngữ cảnh $s$ về một mã khái niệm $c^* \in \mathcal{K}$.
- Tách bạch 3 tầng có thể thay thế độc lập: **feedback generation** (LLM), **retrieval** (encoder + vector index), **fusion** (vector interpolation hoặc RRF).
- Cho phép bật/tắt GRF mà không đổi phần còn lại của pipeline (điều kiện để so sánh baseline vs. GRF trong Bảng 1–3).

## 2. Sơ đồ tổng thể

```text
                    ┌───────────────────────────────────────────────┐
                    │            Offline: KB Indexing               │
                    │                                               │
  KB (người dùng)   │  1. Parse concept (CUI, name, aliases)        │
  ─────────────────▶│  2. Homonym disambiguation (alias + (name))   │
                    │  3. Encode từng surface form bằng E(·)        │
                    │  4. Ghi (vector, CUI, surface form) vào index │
                    └──────────────────────┬────────────────────────┘
                                           │  Vector Index I
                                           ▼
  ┌─────────────┐   (m, s)   ┌──────────────────────────┐
  │ Input Layer │───────────▶│  Orchestrator / Pipeline │
  └─────────────┘            └────────────┬─────────────┘
                                          │
              ┌───────────────────────────┼───────────────────────────┐
              ▼                           ▼                           ▼
     ┌─────────────────┐        ┌──────────────────┐        ┌──────────────────┐
     │ Query Encoder   │        │ Feedback         │        │ Feedback         │
     │ E(·) → Q_0      │        │ Generator (LLM)  │        │ Generator (LLM)  │
     └────────┬────────┘        │ → F_1 … F_n      │        │ → F_i            │
              │                 └────────┬─────────┘        └────────┬─────────┘
              │                          │ Q_Fi = E(F_i)             │
              │                          ▼                           │
              │                 ┌──────────────────┐                 │
              │                 │  Fusion Layer    │◀────────────────┘
              │                 │  • Vector (α)    │
              │                 │  • Rank (RRF)    │
              │                 └────────┬─────────┘
              │                          ▼
              │                 ┌──────────────────┐
              └────────────────▶│  Retrieval       │
                                │  Cosine / Search │
                                └────────┬─────────┘
                                         ▼
                                ┌──────────────────┐
                                │ Concept-level    │
                                │ Aggregation      │
                                └────────┬─────────┘
                                         ▼
                                ┌──────────────────┐
                                │ Re-ranker (LLM)  │
                                │ hoặc Pick Top-1  │
                                └────────┬─────────┘
                                         ▼
                                   c* (CUI)
```

## 3. Thành phần (components)

| # | Thành phần | Trách nhiệm | Thay thế được bằng |
| --- | --- | --- | --- |
| C1 | KB Loader / Parser | Đọc KB do người dùng cung cấp, tạo bản ghi `(c, N_c, A_c)` | Định dạng JSONL do người dùng cung cấp |
| C2 | Homonym Resolver | Khử trùng tên: `a + " (" + N_c + ")"` | Quy tắc chuỗi, hoặc embedding-based |
| C3 | Encoder $E(\cdot)$ | Nhúng mention (pool token mention trong câu, xem [encoder](02-encoder.md)), feedback và surface form thành vector | BioSyn, SapBERT |
| C4 | Vector Index $\mathcal{I}$ | Lưu `(vector, CUI, surface form)`, tìm cosine top-k | [retrieval](03-retrieval.md) — exact cosine trên `pycauset`, KB memory-mapped |
| C5 | Feedback Generator | Gọi LLM sinh $F_{\text{def}}, F_{\text{syn}}, F_{\text{std}}$ | GPT-4o, LLM nội bộ |
| C6 | Fusion | Gộp $Q_0$ với $Q_{F_i}$ (pre) hoặc gộp ranked list (post) | Vector interp., Weighted RRF |
| C7 | Concept Aggregator | Gộp điểm surface form → điểm concept | `max` trên alias |
| C8 | Re-ranker | Xếp hạng lại top-k concept | GPT-4o, heuristic top-1 |
| C9 | Evaluator | Tính acc@k và accuracy end-to-end | Script đánh giá |

## 4. Tài liệu con

| Tài liệu | Nội dung |
| --- | --- |
| [data-contracts](01-data-contracts.md) | §4 — bản ghi KB, entry index, Request/Response, mention luôn nằm trong ngữ cảnh |
| [encoder](02-encoder.md) | C3 — $Q_0$ pool token mention trong câu, quy tắc chuẩn hoá |
| [retrieval](03-retrieval.md) | C4 + §5.5 + §7 — index, `pycauset.matmul`, KB memory-mapped, top-k, aggregation |
| [feedback](04-feedback.md) | C5 + §6 — prompt, ba loại feedback, All-in-One |
| [fusion](05-fusion.md) | C6 + §5.2–5.4 + §7.2/7.3/7.6 — vector interpolation và RRF |
| [reranking](06-reranking.md) | C8 + §8 — pick top-1 và LLM re-ranker |
| [configuration](07-configuration.md) | §9–11 — tham số, ma trận thực nghiệm, đo lường, điểm mở rộng |
| [matmul](08-matmul.md) | C4 — phép nhân `R = Q·Aᵀ`: `pycauset` trên CPU, torch theo block trên GPU, top-k trên block |

Bố cục file trên đĩa (artifact, output) xem [`../storage-layout.md`](../storage-layout.md).
