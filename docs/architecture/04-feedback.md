# C5 — Feedback Generator (LLM)

Thuộc [kiến trúc GRF](README.md). Vector của feedback do [encoder](02-encoder.md)
tạo; việc gộp chúng nằm ở [fusion](05-fusion.md).

## Khung prompt (prompt template)

```text
Context: {s}
Mention: {m}
Instruction: {t}
```

## Ba loại feedback

| Loại feedback | Ký hiệu | Nguồn | Tham số | Vai trò kiến trúc |
| --- | --- | --- | --- | --- |
| Entity Definition | $F_{\text{def}}$ | Shlyk et al. (2024) | — | Bổ sung ngữ nghĩa, giảm lexical gap |
| N-Synonyms | $F_{\text{syn}}$ | Dobbins (2024) | $n \in \{1,3,5,10\}$ | Mở rộng từ vựng; vector là trung bình các synonym |
| Standard Name | $F_{\text{std}}$ | Đề xuất của bài báo | tên KB đích | Ánh xạ về danh pháp chuẩn |

Cả ba loại đều sinh **văn bản chữ tự nhiên**; không loại nào trả về CUI hay mã
khái niệm. $F_{\text{std}}$ chỉ là một chuỗi tên chuẩn, vẫn đi qua encoder như
mọi feedback khác — nó không phải một phép tra KB.

`feedback_types` nhận từng loại đơn lẻ hoặc cả ba. Bật cả ba là **All-in-One**
(paper §2.3.3): mỗi loại là một component độc lập, trọng số $1/(n+1)$ với $n=3$,
tức $1/4$ cho $Q_0$ và mỗi $Q_{F_i}$ — dùng đúng công thức ở
[fusion §7.2](05-fusion.md#72-pha-2a-vector-interpolation-pre-retrieval),
không có đường mã riêng. Đây là cấu hình đạt đỉnh trong phần lớn bảng của bài báo.

## Ghi chú triển khai

- Gọi song song các loại feedback; cache theo `(m, s, t)` để tránh gọi lặp.
- Với $F_{\text{syn}}$: $\vec{Q}_F = \frac{1}{n}\sum_{j=1}^{n} E(F_{\text{syn},j})$ —
  cần parse danh sách trước khi encode.
- Timeout/fallback: nếu LLM lỗi, pipeline thoái hóa về $Q_0$ (tức baseline).
- Feedback sinh **một lần** cho cả bundle (`GRF/feedback.ipynb`), ghi ra một
  JSONL; mọi kịch bản trong `ranking.ipynb` replay bundle theo khóa `kind`.
