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
