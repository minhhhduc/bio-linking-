# C8 — Re-ranker

Thuộc [kiến trúc GRF](README.md). Đầu vào là danh sách candidate do
[retrieval](03-retrieval.md) trả về.

- **Pick Top-1:** lấy CUI có `Score(c)` lớn nhất — không phụ thuộc LLM, độ trễ
  thấp.
- **LLM Re-ranker:** nhận top-$k$ ứng viên (kèm tên/định nghĩa) và $s$, trả về
  CUI. Thực nghiệm dùng GPT-4o với $k \in \{2,3,5,10\}$.

Ranh giới kiến trúc: re-ranker **không** được truy cập index; chỉ làm việc trên
danh sách ứng viên.

## Đo lường end-to-end

Accuracy sau re-ranking (Bảng 3): tỉ lệ mention mà CUI cuối cùng sau re-rank bằng
gold CUI. Kết quả ghi vào `e2e_results/` — xem
[configuration §10](07-configuration.md#10-đo-lường).

## acc@k sau re-rank — đọc số này thế nào

Hiện tại LLM trả về **một** CUI (`_parse_answer` từ chối câu trả lời chứa nhiều
CUI). Danh sách sau re-rank được dựng bởi `placed()` trong
[`source/GRF/candidates.py`](../../source/GRF/candidates.py):

```text
re-ranked list = [CUI do LLM chọn] + (các ứng viên còn lại, giữ thứ tự retrieval)
```

`accuracy_at_k` tính trên đúng danh sách đó, nên:

- **`accuracy` (và `acc@1`)**: tỉ lệ LLM chọn đúng gold — con số thật của
  re-ranker, dùng làm số chính.
- **`acc@k`, k > 1**: "retrieval top-k bị chiếm một suất bởi pick của LLM".
  Nếu pick sai và nằm ngoài top-k, nó đẩy hạng k của retrieval xuống k+1 nên
  số này có thể **giảm** so với retrieval thuần. Đây không phải thứ hạng
  top-k do LLM tạo.
- Số liệu nằm ở `.../reranking/*_e2e.json`: `accuracy`, `acc@k`, `by_label`.
  Retrieval thuần nằm ở `.../ranking/.../*.json`. Thứ tự sau re-rank **không**
  được lưu per-mention, nên không thể tính lại offline.

### Nếu sau này cần acc@k do LLM xếp thật (k > 1)

Sửa đúng 3 chỗ:

1. [`source/GRF/prompts.yaml`](../../source/GRF/prompts.yaml) — mục
   `rerank.instruction`: yêu cầu trả danh sách CUI theo thứ tự, mỗi dòng một CUI.
2. [`source/GRF/rerank.py`](../../source/GRF/rerank.py) — `_parse_answer`
   (nhận list có thứ tự, bỏ CUI lạ/trùng) và `rerank_batch` (kiểu trả về thành
   `list[str]`).
3. [`source/GRF/candidates.py`](../../source/GRF/candidates.py) — `placed()`:
   dựng list theo thứ tự model trả, phần còn lại nối sau theo retrieval.

Fallback giữ nguyên: câu trả lời không dùng được → giữ thứ tự retrieval cho
mention đó. Chạy lại **chỉ stage rerank** (đọc candidate JSONL, không cần
encoder/GPU).
