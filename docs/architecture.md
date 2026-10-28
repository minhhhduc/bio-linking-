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
| C3 | Encoder $E(\cdot)$ | Nhúng mention (pool token mention trong câu, §4.4), feedback và surface form thành vector | BioSyn, SapBERT |
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
Request  { mention: string, context: string, feedback_types: string[], n_syn: int,
           start_token?: int, end_token?: int }   // span token của mention trong context
Response { cui: string, candidates: [{cui, score, surface}], debug: {Q_0, Q_F, ranks} }
```

Bản gold tiếng Anh (cho thí nghiệm song ngữ) không do pipeline này sinh ra: nó
được dịch offline từ gold tiếng Việt, giữ nguyên `cui` / `icd10` / `label` và mọi
field `vi_*`, chỉ thêm `en_sentence_str`, `en_term` và span `en_*` tương ứng —
xem [`translate-gold-pipeline.md`](translate-gold-pipeline.md).

### 4.4. Nguyên tắc chung: mention luôn nằm trong ngữ cảnh

Mỗi mention $m$ luôn đi kèm câu chứa nó $s$ (`Request.context`); không có chế độ
"mention rời". Dense query dùng chung cho **baseline và GRF** giữ nguyên câu và
đánh dấu mention **tại đúng vị trí của nó**:

```text
text = s[:start] + " [SEP] " + m + " [SEP] " + s[end:]
Q    = normalize( E(m) + mean-pool( hidden states của các token nằm giữa hai [SEP] ) )
```

- `start` / `end` là span ký tự của mention trong câu, suy ra từ span token gold
  (`start_token`, `end_token`, token hoá bằng khoảng trắng); nếu thiếu thì định
  vị mention bằng so khớp chuỗi.
- Phần câu **trước** và **sau** mention vẫn là context: tham gia self-attention
  nên vector của mention phụ thuộc ngữ cảnh, nhưng **không** được mean vào
  vector — tránh pha loãng tín hiệu mention.
- `[CLS]` / `[SEP]` (offset 0) bị loại; chỉ token thực giao với span ký tự của
  mention được đưa vào phép mean.
- Truncation câu dài có thể cắt mất mention: encoder báo lỗi thay vì lặng lẽ
  pool nhầm token.
- Vector được L2-normalize trước khi so cosine với index.

Phía index KB vẫn encode surface form độc lập ($E(a)$), vì tên khái niệm không
có ngữ cảnh. Vì vậy $Q$ **cộng thêm $E(m)$** — cùng hàm `encoder.encode` đã dùng
để dựng row — rồi mới normalize: mean-pool trong ngữ cảnh là một hàm khác, vector
của nó lệch khỏi đúng row mà nó phải khớp. Baseline và GRF dùng cùng một hàm
(`SentenceTransformerEncoder.encode_mention_in_context`) cho $Q_0$, nên hai bên
retrieve trên cùng một biểu diễn và số liệu so sánh được.

Hai notebook `ranking` (baseline và GRF) có công tắc `USE_CONTEXT_QUERY`: `True` dùng công thức
trên; `False` bỏ ngữ cảnh và lấy $Q = E(m)$ bằng chính `encoder.encode` của thư
viện — đúng hàm đã dựng index row.

## 5. Luồng xử lý

### 5.1. Offline — build index

1. Load KB → danh sách concept.
2. Với mỗi alias $a \in \mathcal{A}_c$: nếu $a$ xuất hiện ở nhiều concept, thay bằng $a + \text{" ("} + \mathcal{N}_c + \text{")"}$.
3. Encode từng surface form bằng $E(\cdot)$.
4. Ghi entry vào $\mathcal{I}$ (kèm metadata để lọc theo KB đích).

### 5.2. Online — Pipeline 1: Vector-based GRF (pre-retrieval)

```text
m, s ──▶ E_ctx(m, s) = Q_0           // §4.4: pool token của mention trong câu
     ──▶ LLM(m, s, t_i) = F_i        (song song cho từng t_i)
     ──▶ E(F_i) = Q_Fi
     ──▶ Q_GRF = w_0·Q_0 + Σ w_i·Q_Fi      (w_0 = w_i = 1/(n+1), §7.2)
     ──▶ cosine(Q_GRF, I) → top-k surface forms
     ──▶ score(c) = max_{a ∈ A_c ∩ top-k} S(a)
     ──▶ top-k concept → re-ranker / pick top-1
```

Đặc điểm: **chỉ một lần quét index** — chi phí retrieval không tăng theo số loại feedback, chỉ tăng ở bước encode $F_i$.

Cấu hình trọng số:

- Đơn lẻ ($n = 1$): $w_0 = w_1 = \frac{1}{n+1} = 0.5$.
- All-in-One ($n$ loại): mỗi thành phần trọng số $\frac{1}{n+1}$.

Toàn bộ batch được xử lý theo các pha ma trận độc lập ở §7, không có vòng lặp Python cho từng mention.

### 5.3. Online — Pipeline 2: Rank-based GRF (post-retrieval)

```text
m, s ──▶ E_ctx(m, s) = Q_0     ──▶ Search(Q_0, I)  = R_0
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

### 5.5. Chấm điểm hàng loạt

Cả hai pipeline dùng chung một phép retrieval; chỉ vector query là khác:

```text
A ∈ R^{n_surface × d}      # ma trận index, mỗi hàng một surface form
Q ∈ R^{b × d}              # b query của một batch
R  = Q · A^T               # (b × n_surface) cosine, vì mọi hàng đã unit-norm
top-k theo từng hàng của R → candidate của mention đó
```

`VectorIndex.search_batch` thực hiện đúng phép này. Khi `use_gpu()` bật, phép
nhân và `topk` chạy trên GPU theo chunk query; chỉ các hàng đạt top-k mới được
copy về CPU, nên không có vòng lặp Python cho từng mention. Baseline encode hết
query rồi chấm điểm một lần cho cả tập; GRF encode toàn bộ `Q_0` trong
`link_batch` bằng một lượt forward, sau đó sinh feedback và chấm điểm cho từng
request.

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

## 7. Retrieval — các pha độc lập và vector hóa

Retrieval gồm các pha ma trận độc lập; không có vòng lặp Python cho từng mention hay surface form.

### 7.1. Pha 1: Encode queries (batch)

$$
\begin{aligned}
Q_0 &\in \mathbb{R}^{b \times d} && \text{(} b \text{ mention, một lượt forward)} \\
F &\in \mathbb{R}^{b \times n \times \ell} && \text{(feedback texts, song song per request)} \\
Q_F &\in \mathbb{R}^{b \times n \times d} && \text{(encode toàn bộ feedback, một lượt)}
\end{aligned}
$$

- $b$: batch size (số mention)
- $n$: số feedback component thành công per request
- $d$: encoder dimension
- $\ell$: độ dài text (biến thiên)

### 7.2. Pha 2a: Vector interpolation (pre-retrieval)

$$
\begin{aligned}
w_0 &= \frac{1}{n+1}, \quad w_i = \frac{1}{n+1} \quad \forall i \in [1, n] \\
Q_{\text{GRF}}[j] &= w_0 \cdot Q_0[j] + \sum_{i=1}^{n_j} w_i \cdot Q_F[j, i] && \text{per mention } j \\
Q_{\text{GRF}} &= \text{L2-normalize}(Q_{\text{GRF}}) \in \mathbb{R}^{b \times d}
\end{aligned}
$$

### 7.3. Pha 2b: RRF retrieval (post-retrieval)

$$
\begin{aligned}
Q_{\text{all}}[j] &= [Q_0[j]; Q_F[j, 1]; \ldots; Q_F[j, n_j]] && \in \mathbb{R}^{(n_j+1) \times d} \\
\text{RRF}(c; j) &= \sum_{x=0}^{n_j} w_x \cdot \frac{1}{r_x(c)} && \text{where } \frac{1}{r_x(c)} = 0 \text{ if } c \notin R_x
\end{aligned}
$$

Với mọi $j$, scan index $(n_j + 1)$ lần song song (batch); mỗi query trả về rank list $R_x$.

### 7.4. Pha 3: Matmul + top-k (vectorized)

$$
\begin{aligned}
A &\in \mathbb{R}^{N \times d} && \text{(index matrix, } N \text{ surface forms)} \\
Q &\in \mathbb{R}^{m \times d} && \text{(queries: } Q_{\text{GRF}} \text{ hoặc } Q_{\text{all}} \text{ flat)} \\
S &= Q \cdot A^\top \in \mathbb{R}^{m \times N} && \text{(cosine scores, cả 2 bên unit-norm)} \\
\text{top-}k(S[i, :]) &\to \{(a, s_{ia})\}_{a \in \text{Top-}k} && \text{per query row } i
\end{aligned}
$$

Quy tắc thứ tự: $(-s, \text{cui}, \text{surface})$ lexicographic; deterministic ngay cả khi điểm bằng nhau.

GPU path: chunk query thành $c$ hàng, $S_{\text{chunk}} = Q_{\text{chunk}} \cdot A^\top$ trên device, `torch.topk(S_chunk, k, dim=0)` per column, copy về CPU chỉ top-$k$ row per query.

### 7.5. Pha 4: Concept aggregation (vectorized)

$$
\begin{aligned}
\text{Score}(c) &= \max_{a \in \mathcal{A}_c \cap \text{Top-}k} S(a) \\
\text{Surface}_{\text{rep}}(c) &= \arg\max_{a \in \mathcal{A}_c \cap \text{Top-}k} \big( S(a), -\text{lex}(a) \big)
\end{aligned}
$$

Triển khai: một lượt `np.lexsort` trên khoá $(-\text{score}, \text{cui}, \text{surface})$ cho toàn bộ block $(m \times k)$, rồi cắt `limit` phần tử đầu của từng hàng bằng `np.bincount` + `cumsum` (không có vòng lặp Python per mention). Tie-break surface: lex ascending.

### 7.6. Pha 5: RRF fusion (nếu dùng pipeline 2)

$$
\begin{aligned}
r_x(c) &= \text{rank của } c \text{ trong } R_x \quad \text{(1-based, hoặc } \infty \text{)} \\
\text{RRF}(c) &= \sum_{x=0}^{n} w_x \cdot \frac{1}{r_x(c)} && w_x = \frac{1}{n+1}
\end{aligned}
$$

Rank list $R_x$ được tính từ hits của query $x$ bằng aggregation pha 5, rồi assign rank 1-based. Fusion là **một** scatter-add: dựng mảng phẳng `owner * n_concepts + cui_id` và cộng $w_x / r_x(c)$ bằng `np.bincount`, thu được ma trận điểm $(b \times |\mathcal{C}|)$ trong một lượt thay vì một dict cho mỗi query.

### 7.7. Sắp xếp cuối và top-k concept

$$
\text{Candidates} = \text{argsort}_{c} \big( -\text{Score}(c), c \big)[:k_{\text{concept}}]
$$

Trả về danh sách `(cui, score, surface_rep)` đã sắp xếp; đây là đầu vào của re-ranker (stage riêng biệt, không thuộc retrieval).

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
- **Candidate accuracy:** $\text{acc@}k = \frac{1}{n}\sum_i \mathbb{1}[\text{gold}_i \in \text{top-}k\text{ candidates}_i]$
  trên danh sách candidate cuối (sau re-ranking nếu có).
- **Đối chứng:** mỗi cấu hình GRF phải chạy cùng seed/model với baseline không GRF để báo cáo "Mức tăng" có ý nghĩa.

## 11. Điểm mở rộng

- Thêm loại feedback mới = thêm một instruction template + một trọng số; không đổi retrieval.
- Thay encoder = thay C3, giữ nguyên C4–C8 (miễn cùng số chiều vector).
- Thay index (FAISS → HNSW) = thay C4; hợp đồng `top-k (score, cui, surface)` giữ nguyên.
- Chạy ablation bằng cách đặt `feedback_types = []` → thoái hóa về dense retrieval thuần.
