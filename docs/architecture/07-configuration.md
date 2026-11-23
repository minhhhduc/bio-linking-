# Cấu hình, thực nghiệm, đo lường

Thuộc [kiến trúc GRF](README.md).

## 9. Cấu hình và tham số

| Tham số | Giá trị | Ghi chú |
| --- | --- | --- |
| `encoder` | `biosyn` \| `sapbert` | ảnh hưởng trực tiếp Bảng 1 vs Bảng 2 |
| `feedback_types` | `()`, `("def",)`, `("syn",)`, `("std",)`, hoặc cả ba | `()` = baseline; một phần tử = cấu hình đơn lẻ; cả ba = All-in-One |
| `n_syn` | 1, 3, 5, 10 | số synonym mỗi lần sinh |
| `alpha` | $1/(n+1)$ | suy ra từ số component, không phải tham số nhập vào |
| `top_k_retrieval` | 10 | số surface form lấy ra |
| `top_k_concept` | 1–10 | số ứng viên đưa vào re-ranker |
| `fusion` | `vector` \| `rrf` | chọn pipeline 1 hoặc 2 |
| `rerank` | `top1` \| `llm` | bật/tắt re-ranking |

### 9.1. Ma trận thực nghiệm (theo §3–4 của bài báo)

**Một lần chạy `ranking.ipynb` đi qua toàn bộ ma trận dưới đây**, không phải
một nhánh mã: hằng số `FEEDBACK_SCENARIOS` và `FUSIONS` liệt kê các cặp cần
thử, và Stage 7 tự dựng một `PipelineConfig` cho từng cặp, lưu candidate ra
`candidates/grf_<scenario>_n<n>_k<k>_candidates.jsonl`, rồi ghi `acc@k` của
mọi kịch bản vào `grf_retrieval_summary.json`. `reranking.ipynb` đọc mọi file
candidate đó và ghi một JSON end-to-end cho mỗi kịch bản vào `e2e_results/`.

`feedback.ipynb` cũng chỉ cần chạy **một lần** với
`FEEDBACK_TYPES = ("def", "syn", "std")`: bundle chứa đủ ba loại, và ranking
chọn tập con theo khóa `kind` của từng record. Ngoại lệ là `n_syn`: $n$ đã nằm
trong câu chỉ thị nên đổi $n$ phải sinh lại `syn` (cache theo prompt nên chỉ
phần `syn` mới decode). Các dòng `encoder` và `LLM_MODEL_LOCAL` bên dưới vẫn
cần đổi hằng số rồi chạy lại, vì chúng đổi cả KB artifact lẫn bundle.

| Kịch bản | Cấu hình |
| --- | --- |
| Baseline, không GRF | `feedback_types=()` (chọn rỗng từ bundle) |
| Single definition | `feedback_types=("def",)` |
| Single N-synonyms, $n \in \{1,3,5,10\}$ | `feedback_types=("syn",)`, `n_syn=n` |
| Single standard name | `feedback_types=("std",)`, `target_kb` đặt theo KB |
| All-in-One ($n_{\text{syn}}=10$) | `feedback_types=("def","syn","std")`, `n_syn=10` |
| Vector-based / Rank-based | `fusion="vector"` / `fusion="rrf"` |
| BioSyn / SapBERT | `encoder` và encoder của KB artifact |
| Độ nhạy mô hình sinh | `LLM_MODEL_LOCAL` (Qwen3 14B/4B/1.7B, Gemma-3 12B/4B/1B) |
| Re-ranker $k$ | `top_k_concept ∈ {2,3,5,10}`, `rerank="llm"` |

Hai phân tích của bài báo **không** được triển khai ở đây, và không nên báo cáo
như thể có:

- **Text-based GRF** (paper §2.3.2: nối mention + feedback thành một chuỗi rồi
  encode). Repo chỉ có vector-based và rank-based.
- **Quét $\alpha$**: $\alpha = 1/(n+1)$ là cố định, không phải tham số cấu hình.

## 10. Đo lường

- **Candidate retrieval:** $\text{acc@}k = \frac{1}{n}\sum_i \mathbb{1}[\text{gold}_i \in \text{top-}k\text{ candidates}_i]$
  trên danh sách candidate `ranking.ipynb` lưu ra, với $k \le$ `top_k_concept`.
  Đây là thang đo duy nhất cho phần retrieval; repo không báo cáo R@k.
- **End-to-end normalization:** accuracy sau re-ranking (Bảng 3): tỉ lệ mention
  mà CUI cuối cùng sau re-rank bằng gold CUI.
- **Đối chứng:** mỗi cấu hình GRF phải chạy cùng seed/model với baseline không GRF để báo cáo "Mức tăng" có ý nghĩa.

## 11. Điểm mở rộng

- Thêm loại feedback mới = thêm một instruction template + một trọng số; không đổi retrieval.
- Thay encoder = thay C3, giữ nguyên C4–C8 (miễn cùng số chiều vector).
- Thay index (exact cosine → FAISS/HNSW) = thay C4; hợp đồng `top-k (score, cui, surface)` giữ nguyên.
- Chạy ablation bằng cách đặt `feedback_types = []` → thoái hóa về dense retrieval thuần.
