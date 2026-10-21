## 1. Định nghĩa bài toán (Problem Formulation)

Biomedical Entity Linking (BEL) là bài toán chuẩn hóa một thực thể y sinh được nhắc đến trong văn bản về một khái niệm duy nhất trong cơ sở tri thức (Knowledge Base - KB).

- **Hiện tượng thắt cổ chai:** Các mô hình truy xuất mật độ cao (Dense Retrieval) bị giới hạn trần hiệu năng (gọi là *bounded recall*) do khoảng cách từ vựng (lexical gap), từ viết tắt không chuẩn hóa, lỗi chính tả, hoặc sự phân hóa thuật ngữ giữa ngôn ngữ lâm sàng tự nhiên và danh pháp chuẩn trong KB.
- **Mục tiêu:** Sử dụng Generative Relevance Feedback (GRF) từ mô hình ngôn ngữ lớn (LLM) để bổ sung ngữ cảnh, mở rộng truy vấn và tối ưu hóa việc truy xuất ứng viên (Candidate Retrieval) cũng như xếp hạng cuối (Re-ranking).

## 2. Đầu vào và Đầu ra (Input & Output)

**Đầu vào (Input):**

- Chuỗi thực thể cần chuẩn hóa: $m$ (Mention span).
- Câu văn ngữ cảnh chứa thực thể: $s$ (Context sentence).
- Cơ sở tri thức: $\mathcal{K} = \{(c, \mathcal{N}_c, \mathcal{A}_c)\}$, trong đó $c$ là mã định danh thực thể (CUI), $\mathcal{N}_c$ là tên chuẩn hóa (preferred name), $\mathcal{A}_c$ là tập hợp tất cả các tên gọi thay thế/từ đồng nghĩa (aliases/surface forms).

**Đầu ra (Output):**

- Mã khái niệm chuẩn duy nhất: $c^* \in \mathcal{K}$.

## 3. Phương pháp xử lý trùng tên (Homonym Disambiguation)

Trong KB y sinh, hiện tượng từ đồng âm xảy ra khi một tên gọi (alias) $a$ trỏ đồng thời về hai hoặc nhiều khái niệm khác nhau: $a \in \mathcal{A}_{c_1} \cap \mathcal{A}_{c_2}$ (ví dụ: cụm "conorenal syndrome" vừa trỏ về MESH:D012779, vừa trỏ về MESH:C535463).

Quy tắc ghép chuỗi khử đồng âm:

$$
a_{\text{disambiguated}} = a + \text{" ("} + \mathcal{N}_c + \text{")"}
$$

- Chuỗi 1: "conorenal syndrome (short rib–polydactyly syndrome)" $\rightarrow c_1 = \text{MESH:D012779}$
- Chuỗi 2: "conorenal syndrome (Mainzer–Saldino disease)" $\rightarrow c_2 = \text{MESH:C535463}$

**Cơ chế lưu trữ Index:** Mỗi tên gọi (cả preferred name và alias đã khử trùng) đều được nhúng độc lập thành một vector $\vec{v} \in \mathbb{R}^d$ bằng mô hình encoder (BioSyn hoặc SapBERT) và lưu kèm mã định danh $c$ vào cơ sở dữ liệu vector. Một khái niệm $c$ có bao nhiêu alias thì sẽ có bấy nhiêu vector đại diện độc lập trong cơ sở dữ liệu.

## 4. Phương pháp Generative Relevance Feedback (GRF)

Hệ thống sử dụng LLM để sinh văn bản phản hồi $F$ dựa trên thông tin thực thể và ngữ cảnh văn bản.

**Khung Prompt tổng quát (Prompt Template)**

```text
Context: {s}
Mention: {m}
Instruction: {t}
```

**Nội dung chỉ thị $\{t\}$ cho từng loại phản hồi**

- **Entity Definition ($F_{\text{def}}$):**
  Yêu cầu: "Dựa vào ngữ cảnh {s}, hãy đưa ra một định nghĩa ngắn gọn trong một câu để giải thích chính xác ý nghĩa của thực thể {m}."
- **N-Synonyms ($F_{\text{syn}}$):**
  Yêu cầu: "Dựa vào ngữ cảnh {s}, hãy liệt kê đúng {n} từ đồng nghĩa hoặc các cách diễn đạt tương đương (alternative phrasings) cho thực thể {m}." (trong đó $n \in \{1, 3, 5, 10\}$).
- **Standard Name ($F_{\text{std}}$):**
  Yêu cầu: "Dựa vào ngữ cảnh {s}, hãy tạo ra tên chuẩn học thuật được cộng đồng công nhận (standard community-recognized name) cho thực thể {m}, tuân theo hệ danh pháp của {tên KB đích, ví dụ: NCBI GENE, NCBI TAXONOMY, MeSH}."

## 5. Kiến trúc Pipeline

### Pipeline 1: Vector-based GRF (Gộp trước khi truy xuất - Pre-retrieval)

```text
[Mention m, Context s]
       │
       ├──────────────────────────────┬──────────────────────────────┐
       ▼                              ▼                              ▼
  Encoder E(·)                 LLM Feedback Gen               LLM Feedback Gen
       │                              │                              │
   Vector Q_0                     Feedback F_1                   Feedback F_2
       │                              │                              │
       │                         Encoder E(·)                   Encoder E(·)
       │                              │                              │
       │                          Vector Q_F1                    Vector Q_F2
       │                              │                              │
       └──────────────────────────────┴───────────────┬──────────────┘
                                                      │
                                                      ▼
                                       Linear Interpolation (Weighted Sum)
                                                      │
                                                      ▼
                                            Fused Query Vector Q_GRF
                                                      │
                                                      ▼
                                        Cosine Retrieval (1 lần quét DB)
                                                      │
                                                      ▼
                                         Top-k Surface Forms / Aliases
                                                      │
                                                      ▼
                                         Concept-level Aggregation
                                                      │
                                                      ▼
                                          Top-k Candidate Concepts (CUI)
                                                      │
                                                      ▼
                                        LLM Re-ranker (hoặc Pick Top-1)
                                                      │
                                                      ▼
                                            Final Concept CUI c*
```

**Bước 1 - Sinh Feedback:** LLM nhận $(m, s, t)$ và tạo ra văn bản $F$.

**Bước 2 - Mã hóa vector:**

$$
\vec{Q}_0 = E(m), \quad \vec{Q}_F = E(F)
$$

(Nếu $F$ là danh sách gồm $n$ từ đồng nghĩa, $\vec{Q}_F = \frac{1}{n}\sum_{j=1}^n E(F_{\text{syn}, j})$).

**Bước 3 - Nội suy vector:**

Cấu hình đơn lẻ:

$$
\vec{Q}_{\text{GRF}} = \alpha \vec{Q}_0 + (1 - \alpha) \vec{Q}_F \quad \left(\text{với } \alpha = \frac{1}{n+1}, \text{ tức } \alpha = 0.5 \text{ khi } n = 1\right)
$$

Cấu hình All-in-One ($n$ loại feedback):

$$
\vec{Q}_{\text{GRF}} = \frac{1}{n+1} \vec{Q}_0 + \sum_{i=1}^n \frac{1}{n+1} \vec{Q}_{F_i}
$$

**Bước 4 - Truy xuất:** Tính khoảng cách Cosine giữa $\vec{Q}_{\text{GRF}}$ và toàn bộ các vector tên gọi trong KB $\mathcal{I}$:

$$
S(a) = \cos(\vec{Q}_{\text{GRF}}, \vec{v}_a) = \frac{\vec{Q}_{\text{GRF}} \cdot \vec{v}_a}{\Vert{}\vec{Q}_{\text{GRF}}\Vert{} \Vert{}\vec{v}_a\Vert{}}
$$

**Bước 5 - Gộp mức Concept (Concept-level Aggregation):** Với mỗi khái niệm $c$, điểm số đại diện là điểm lớn nhất trong số các aliases của nó lọt vào tập kết quả:

$$
\text{Score}(c) = \max_{a \in \mathcal{A}_c \cap \text{Top-}k} S(a)
$$

**Bước 6 - Xếp hạng & Xuất kết quả:** Chọn CUI đứng đầu hoặc chuyển danh sách ứng viên cho LLM Re-ranker.

### Pipeline 2: Rank-based GRF (Gộp sau khi truy xuất - Post-retrieval)

```text
[Mention m, Context s]
       │
       ├──────────────────────────────┬──────────────────────────────┐
       ▼                              ▼                              ▼
  Encoder E(·)                 LLM Feedback Gen               LLM Feedback Gen
       │                              │                              │
   Vector Q_0                     Feedback F_1                   Feedback F_2
       │                              │                              │
  Cosine Search                  Encoder E(·)                   Encoder E(·)
  trên KB Index                       │                              │
       │                          Vector Q_F1                    Vector Q_F2
       │                              │                              │
       │                         Cosine Search                  Cosine Search
       │                         trên KB Index                  trên KB Index
       │                              │                              │
       ▼                              ▼                              ▼
   Ranked List R_0                Ranked List R_F1               Ranked List R_F2
       │                              │                              │
       └──────────────────────────────┴───────────────┬──────────────┘
                                                      │
                                                      ▼
                                       Weighted Reciprocal Rank Fusion (RRF)
                                                      │
                                                      ▼
                                            Fused Score RRF(c)
                                                      │
                                                      ▼
                                          Top-k Candidate Concepts (CUI)
                                                      │
                                                      ▼
                                            Final Concept CUI c*
```

**Bước 1 - Truy xuất song song:** Thực hiện truy vấn độc lập cho vector gốc $\vec{Q}_0$ và từng vector feedback $\vec{Q}_{F_i}$ trên chỉ mục KB:

$$
R_0 = \text{Search}(\vec{Q}_0), \quad R_F = \text{Search}(\vec{Q}_F)
$$

**Bước 2 - Hợp nhất thứ hạng (Weighted RRF):** Tính điểm tổng hợp cho từng ứng viên $c$ dựa trên thứ hạng (vị trí đứng $r(c)$) trong các danh sách:

Cấu hình đơn lẻ:

$$
\text{RRF}(c) = \frac{\alpha}{r_0(c)} + \frac{1 - \alpha}{r_F(c)} \quad \left(\text{với } \alpha = \frac{1}{n+1}, \text{ tức } \alpha = 0.5 \text{ khi } n = 1\right)
$$

Cấu hình All-in-One ($n$ loại feedback):

$$
\text{RRF}(c) = \frac{1}{n+1} \cdot \frac{1}{r_0(c)} + \sum_{i=1}^n \frac{1}{n+1} \cdot \frac{1}{r_{F_i}(c)}
$$

(Nếu khái niệm $c$ không xuất hiện trong danh sách $R_x$, đặt $\frac{1}{r_x(c)} = 0$).

**Bước 3 - Xếp hạng lại:** Sắp xếp toàn bộ ứng viên theo điểm $\text{RRF}(c)$ giảm dần để lấy Top-$k$ ứng viên cho bước re-ranking.

## 6. Kết quả thực nghiệm

### Bảng 1: Hiệu năng Truy xuất ứng viên trên mô hình BioSyn (Table 3)

| Tập dữ liệu | Tiêu chí | Baseline | GRF Tốt nhất | Mức tăng | Cấu hình đạt đỉnh |
| --- | --- | ---: | ---: | ---: | --- |
| NCBI Disease | R@1 | 71.35% | 73.78% | +2.43% | Vector-based (+10 syn) |
| NCBI Disease | R@5 | 85.92% | 90.48% | +4.56% | Rank-based (All-in-One) |
| BC5CDR (Disease) | R@1 | 74.69% | 78.51% | +3.82% | Vector-based (+10 syn) |
| BC5CDR (Disease) | R@5 | 87.19% | 91.97% | +4.78% | Rank-based (All-in-One) |
| BC5CDR (Chemical) | R@1 | 82.01% | 93.87% | +11.86% | Vector-based (All-in-One) |
| BC5CDR (Chemical) | R@5 | 88.65% | 97.55% | +8.90% | Rank-based (All-in-One) |
| NLM-Chem | R@1 | 70.90% | 77.53% | +6.63% | Vector-based (All-in-One) |
| NLM-Chem | R@5 | 79.87% | 88.57% | +8.70% | Rank-based (All-in-One) |
| GNormPlus | R@1 | 69.15% | 79.29% | +10.14% | Rank-based (All-in-One) |
| GNormPlus | R@5 | 85.26% | 93.73% | +8.47% | Rank-based (All-in-One) |
| NLM-Gene | R@1 | 33.80% | 39.28% | +5.48% | Vector-based (All-in-One) |
| NLM-Gene | R@5 | 78.87% | 85.13% | +6.26% | Rank-based (+std name) |
| S800 | R@1 | 61.61% | 77.11% | +15.50% | Vector-based (All-in-One) |
| S800 | R@5 | 75.00% | 87.32% | +12.32% | Rank-based (All-in-One) |
| Linnaeus | R@1 | 69.06% | 84.30% | +15.24% | Vector-based (All-in-One) |
| Linnaeus | R@5 | 86.74% | 95.80% | +9.06% | Rank-based (All-in-One) |

### Bảng 2: Hiệu năng Truy xuất ứng viên trên mô hình SapBERT (Table 3)

| Tập dữ liệu | Tiêu chí | Baseline | GRF Tốt nhất | Mức tăng | Cấu hình đạt đỉnh |
| --- | --- | ---: | ---: | ---: | --- |
| NCBI Disease | R@1 | 65.53% | 67.66% | +2.13% | Vector-based (+10 syn) |
| NCBI Disease | R@5 | 83.00% | 86.40% | +3.40% | Rank-based (All-in-One) |
| BC5CDR (Disease) | R@1 | 70.98% | 73.95% | +2.97% | Vector-based (All-in-One) |
| BC5CDR (Disease) | R@5 | 83.02% | 85.80% | +2.78% | Rank-based (+std name) |
| BC5CDR (Chemical) | R@1 | 81.37% | 92.93% | +11.56% | Vector-based (All-in-One) |
| BC5CDR (Chemical) | R@5 | 86.93% | 96.83% | +9.90% | Rank-based (All-in-One) |
| NLM-Chem | R@1 | 63.60% | 74.78% | +11.18% | Vector-based (All-in-One) |
| NLM-Chem | R@5 | 73.40% | 84.77% | +11.37% | Rank-based (All-in-One) |
| GNormPlus | R@1 | 55.00% | 66.50% | +11.50% | Rank-based (All-in-One) |
| GNormPlus | R@5 | 77.01% | 89.41% | +12.40% | Rank-based (All-in-One) |
| NLM-Gene | R@1 | 26.13% | 30.45% | +4.32% | Rank-based (All-in-One) |
| NLM-Gene | R@5 | 50.39% | 59.62% | +9.23% | Rank-based (All-in-One) |
| S800 | R@1 | 53.87% | 70.49% | +16.62% | Vector-based (All-in-One) |
| S800 | R@5 | 68.66% | 81.76% | +13.10% | Rank-based (All-in-One) |
| Linnaeus | R@1 | 63.53% | 80.44% | +16.91% | Vector-based (All-in-One) |
| Linnaeus | R@5 | 81.76% | 95.13% | +13.37% | Rank-based (All-in-One) |

### Bảng 3: Độ chính xác chuẩn hóa End-to-End với GPT-4o Re-ranker (Table 4)

Hệ thống sử dụng các ứng viên sinh ra từ mô hình BioSyn kết hợp Vector-based GRF All-in-One, so sánh với pipeline cơ sở tương đương không sử dụng GRF:

| Tập dữ liệu | Thiết lập Re-ranking | Độ chính xác (Không GRF) | Độ chính xác (Có GRF) | Mức chênh lệch (+/-) |
| --- | --- | ---: | ---: | ---: |
| NCBI Disease | Pick Top-1 (Không Re-rank) | 71.35% | 74.75% | +3.40% |
| NCBI Disease | Re-rank Top-2 ứng viên | 71.84% | 75.72% | +3.88% |
| NCBI Disease | Re-rank Top-3 ứng viên | 73.30% | 75.72% | +2.42% |
| NCBI Disease | Re-rank Top-5 ứng viên | 73.27% | 74.75% | +1.48% |
| NCBI Disease | Re-rank Top-10 ứng viên | 72.81% | 76.21% | +3.40% |
| BC5CDR (Disease) | Pick Top-1 (Không Re-rank) | 74.69% | 76.69% | +2.00% |
| BC5CDR (Disease) | Re-rank Top-2 ứng viên | 75.61% | 79.01% | +3.40% |
| BC5CDR (Disease) | Re-rank Top-3 ứng viên | 75.77% | 79.78% | +4.01% |
| BC5CDR (Disease) | Re-rank Top-5 ứng viên | 76.08% | 79.32% | +3.24% |
| BC5CDR (Disease) | Re-rank Top-10 ứng viên | 77.16% | 79.32% | +2.16% |
| BC5CDR (Chemical) | Pick Top-1 (Không Re-rank) | 82.01% | 94.00% | +11.99% |
| BC5CDR (Chemical) | Re-rank Top-2 ứng viên | 82.44% | 93.36% | +10.92% |
| BC5CDR (Chemical) | Re-rank Top-3 ứng viên | 85.01% | 93.79% | +8.78% |
| BC5CDR (Chemical) | Re-rank Top-5 ứng viên | 85.86% | 93.36% | +7.50% |
| BC5CDR (Chemical) | Re-rank Top-10 ứng viên | 86.08% | 92.71% | +6.63% |
| NLM-Chem | Pick Top-1 (Không Re-rank) | 70.90% | 77.58% | +6.68% |
| NLM-Chem | Re-rank Top-2 ứng viên | 72.99% | 79.77% | +6.78% |
| NLM-Chem | Re-rank Top-3 ứng viên | 73.61% | 81.85% | +8.24% |
| NLM-Chem | Re-rank Top-5 ứng viên | 75.07% | 82.06% | +6.99% |
| NLM-Chem | Re-rank Top-10 ứng viên | 75.70% | 82.58% | +6.88% |
| GNormPlus | Pick Top-1 (Không Re-rank) | 69.15% | 78.68% | +9.53% |
| GNormPlus | Re-rank Top-2 ứng viên | 75.04% | 82.22% | +7.18% |
| GNormPlus | Re-rank Top-3 ứng viên | 77.99% | 82.71% | +4.72% |
| GNormPlus | Re-rank Top-5 ứng viên | 80.64% | 84.57% | +3.93% |
| GNormPlus | Re-rank Top-10 ứng viên | 81.92% | 85.26% | +3.34% |
| NLM-Gene | Pick Top-1 (Không Re-rank) | 33.80% | 39.12% | +5.32% |
| NLM-Gene | Re-rank Top-2 ứng viên | 56.49% | 61.34% | +4.85% |
| NLM-Gene | Re-rank Top-3 ứng viên | 58.52% | 66.19% | +7.67% |
| NLM-Gene | Re-rank Top-5 ứng viên | 64.31% | 67.44% | +3.13% |
| NLM-Gene | Re-rank Top-10 ứng viên | 65.72% | 69.17% | +3.45% |
| S800 | Pick Top-1 (Không Re-rank) | 61.61% | 76.05% | +14.44% |
| S800 | Re-rank Top-2 ứng viên | 65.84% | 77.46% | +11.62% |
| S800 | Re-rank Top-3 ứng viên | 65.49% | 76.76% | +11.27% |
| S800 | Re-rank Top-5 ứng viên | 67.25% | 75.00% | +7.75% |
| S800 | Re-rank Top-10 ứng viên | 68.30% | 75.00% | +6.70% |
| Linnaeus | Pick Top-1 (Không Re-rank) | 69.06% | 83.42% | +14.36% |
| Linnaeus | Re-rank Top-2 ứng viên | 80.66% | 90.60% | +9.94% |
| Linnaeus | Re-rank Top-3 ứng viên | 83.42% | 89.50% | +6.08% |
| Linnaeus | Re-rank Top-5 ứng viên | 83.97% | 91.16% | +7.19% |
| Linnaeus | Re-rank Top-10 ứng viên | 85.63% | 91.16% | +5.53% |
