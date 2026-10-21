# Kiến trúc hệ thống — GRF cho Biomedical Entity Linking

Tài liệu này mô tả kiến trúc triển khai (implementation architecture) cho phương pháp trong [`paper/reference_abstract.md`](../paper/reference_abstract.md): dùng Generative Relevance Feedback (GRF) từ LLM để mở rộng truy vấn, nâng trần recall của candidate retrieval và cải thiện xếp hạng cuối.

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
| C3 | Encoder $E(\cdot)$ | Nhúng mention/feedback/surface form thành vector | BioSyn, SapBERT |
| C4 | Vector Index $\mathcal{I}$ | Lưu `(vector, CUI, surface form)`, tìm cosine top-k | FAISS / HNSW / exact |
| C5 | Feedback Generator | Gọi LLM sinh $F_{\text{def}}, F_{\text{syn}}, F_{\text{std}}$ | GPT-4o, LLM nội bộ |
| C6 | Fusion | Gộp $Q_0$ với $Q_{F_i}$ (pre) hoặc gộp ranked list (post) | Vector interp., Weighted RRF |
| C7 | Concept Aggregator | Gộp điểm surface form → điểm concept | `max` trên alias |
| C8 | Re-ranker | Xếp hạng lại top-k concept | GPT-4o, heuristic top-1 |
| C9 | Evaluator | Tính R@1, R@5, accuracy end-to-end | Script đánh giá |

## 4. Dữ liệu và hợp đồng giữa các tầng

### 4.1. Bản ghi KB

KB do người dùng cung cấp; kiến trúc không ràng buộc nguồn KB cụ thể.

```text
Concept {
  cui:      string          // mã concept do KB cung cấp
  name:     string          // N_c — preferred name
  aliases:  string[]        // A_c — surface forms
}
```

### 4.2. Bản ghi index

```text
IndexEntry {
  vector:   float[d]        // E(surface_form)
  cui:      string
  surface:  string          // alias hoặc "alias (preferred name)"
}
```

Ràng buộc: **một concept có bao nhiêu alias thì có bấy nhiêu entry độc lập**; không gộp vector theo concept ở tầng index.

### 4.3. Đầu vào / đầu ra pipeline

```text
Request  { mention: string, context: string, feedback_types: string[], n_syn: int }
Response { cui: string, candidates: [{cui, score, surface}], debug: {Q_0, Q_F, ranks} }
```

## 5. Luồng xử lý

### 5.1. Offline — build index

1. Load KB → danh sách concept.
2. Với mỗi alias $a \in \mathcal{A}_c$: nếu $a$ xuất hiện ở nhiều concept, thay bằng $a + \text{" ("} + \mathcal{N}_c + \text{")"}$.
3. Encode từng surface form bằng $E(\cdot)$.
4. Ghi entry vào $\mathcal{I}$ (kèm metadata để lọc theo KB đích).

### 5.2. Online — Pipeline 1: Vector-based GRF (pre-retrieval)

```text
m, s ──▶ E(m) = Q_0
     ──▶ LLM(m, s, t_i) = F_i        (song song cho từng t_i)
     ──▶ E(F_i) = Q_Fi
     ──▶ Q_GRF = α·Q_0 + Σ (1-α_i)·Q_Fi
     ──▶ cosine(Q_GRF, I) → top-k surface forms
     ──▶ score(c) = max_{a ∈ A_c ∩ top-k} S(a)
     ──▶ top-k concept → re-ranker / pick top-1
```

Đặc điểm: **chỉ một lần quét index** — chi phí retrieval không tăng theo số loại feedback, chỉ tăng ở bước encode $F_i$.

Cấu hình trọng số:

- Đơn lẻ ($n = 1$): $\alpha = \frac{1}{n+1} = 0.5$.
- All-in-One ($n$ loại): mỗi thành phần trọng số $\frac{1}{n+1}$.

### 5.3. Online — Pipeline 2: Rank-based GRF (post-retrieval)

```text
m, s ──▶ E(m) = Q_0            ──▶ Search(Q_0, I)  = R_0
     ──▶ LLM(m, s, t_i) = F_i  ──▶ E(F_i) = Q_Fi   ──▶ Search(Q_Fi, I) = R_Fi
     ──▶ RRF(c) = Σ w_x · 1/r_x(c)        (w_x = 1/(n+1))
     ──▶ sort desc → top-k concept → re-ranker
```

Đặc điểm: **n+1 lần quét index** (song song được), bù lại dung hợp theo thứ hạng bền vững hơn với khác biệt thang điểm giữa các query.

### 5.4. Chọn pipeline

| Tiêu chí | Vector-based | Rank-based |
| --- | --- | --- |
| Số lần truy vấn index | 1 | $n+1$ |
| Chi phí encode feedback | có | có |
| Độ bền với lệch thang điểm | trung bình | cao |
| Độ trễ | thấp | cao hơn |
| Khi nào dùng | index lớn, cần nhanh | chất lượng ưu tiên, top-k lớn |

## 6. Tầng prompt (Feedback Generator)

Hợp đồng chung:

```text
Context: {s}
Mention: {m}
Instruction: {t}
```

| Loại feedback | Ký hiệu | Tham số | Vai trò kiến trúc |
| --- | --- | --- | --- |
| Entity Definition | $F_{\text{def}}$ | — | Bổ sung ngữ nghĩa, giảm lexical gap |
| N-Synonyms | $F_{\text{syn}}$ | $n \in \{1,3,5,10\}$ | Mở rộng từ vựng; vector là trung bình các synonym |
| Standard Name | $F_{\text{std}}$ | tên KB đích | Ánh xạ về danh pháp chuẩn |

Ghi chú triển khai:

- Gọi song song các loại feedback; cache theo `(m, s, t)` để tránh gọi lặp.
- Với $F_{\text{syn}}$: $\vec{Q}_F = \frac{1}{n}\sum_{j=1}^{n} E(F_{\text{syn},j})$ — cần parse danh sách trước khi encode.
- Timeout/fallback: nếu LLM lỗi, pipeline thoái hóa về $Q_0$ (tức baseline).

## 7. Tầng fusion

### 7.1. Vector interpolation (pre-retrieval)

```text
Q_GRF = α·Q_0 + (1-α)·Q_F                  # đơn lẻ, α = 1/(n+1)
Q_GRF = 1/(n+1)·Q_0 + Σ 1/(n+1)·Q_Fi       # all-in-one
```

Bất biến: tổng trọng số $= 1$; khi $n=1$ hai công thức trùng nhau.

### 7.2. Weighted RRF (post-retrieval)

```text
RRF(c) = α/r_0(c) + (1-α)/r_F(c)                      # đơn lẻ
RRF(c) = 1/(n+1)·1/r_0(c) + Σ 1/(n+1)·1/r_Fi(c)       # all-in-one
```

Quy ước: nếu $c \notin R_x$ thì $1/r_x(c) = 0$.

### 7.3. Concept aggregation

```text
Score(c) = max_{a ∈ A_c ∩ Top-k} S(a)
```

Lý do dùng `max`: một concept có nhiều alias; chỉ cần một alias khớp tốt là đủ đại diện.

## 8. Tầng re-ranking

- **Pick Top-1:** lấy CUI có `Score(c)` lớn nhất — không phụ thuộc LLM, độ trễ thấp.
- **LLM Re-ranker:** nhận top-$k$ ứng viên (kèm tên/định nghĩa) và $s$, trả về CUI. Thực nghiệm dùng GPT-4o với $k \in \{2,3,5,10\}$.
- Ranh giới kiến trúc: re-ranker **không** được truy cập index; chỉ làm việc trên danh sách ứng viên.

## 9. Cấu hình và tham số

| Tham số | Giá trị | Ghi chú |
| --- | --- | --- |
| `encoder` | `biosyn` \| `sapbert` | ảnh hưởng trực tiếp Bảng 1 vs Bảng 2 |
| `feedback_types` | `def`, `syn`, `std` | tổ hợp tạo nên "All-in-One" |
| `n_syn` | 1, 3, 5, 10 | số synonym mỗi lần sinh |
| `alpha` | $1/(n+1)$ | $n=1 \Rightarrow 0.5$ |
| `top_k_retrieval` | 10 | số surface form lấy ra |
| `top_k_concept` | 1–10 | số ứng viên đưa vào re-ranker |
| `fusion` | `vector` \| `rrf` | chọn pipeline 1 hoặc 2 |
| `rerank` | `top1` \| `llm` | bật/tắt re-ranking |

## 10. Đo lường

- **Candidate retrieval:** R@1, R@5 trên 8 bộ: NCBI Disease, BC5CDR (Disease/Chemical), NLM-Chem, GNormPlus, NLM-Gene, S800, Linnaeus.
- **End-to-end normalization:** accuracy sau re-ranking (Bảng 3).
- **Đối chứng:** mỗi cấu hình GRF phải chạy cùng seed/model với baseline không GRF để báo cáo "Mức tăng" có ý nghĩa.

## 11. Điểm mở rộng

- Thêm loại feedback mới = thêm một instruction template + một trọng số; không đổi retrieval.
- Thay encoder = thay C3, giữ nguyên C4–C8 (miễn cùng số chiều vector).
- Thay index (FAISS → HNSW) = thay C4; hợp đồng `top-k (score, cui, surface)` giữ nguyên.
- Chạy ablation bằng cách đặt `feedback_types = []` → thoái hóa về dense retrieval thuần.
