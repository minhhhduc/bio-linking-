# C4 — Phép nhân ma trận (cosine scan)

Thuộc [kiến trúc GRF](README.md). Đây là tầng duy nhất của retrieval thực sự
tính toán; các bước còn lại (index, lưu trữ, top-k) chỉ xoay quanh nó.

## Công thức

$$ R = Q \cdot A^\top $$

- $A \in \mathbb{R}^{N \times d}$ — KB, mỗi hàng một surface form, đã
  unit-norm.
- $Q \in \mathbb{R}^{b \times d}$ — $b$ query của một batch, đã unit-norm.
- $R \in \mathbb{R}^{b \times N}$ — vì hai vế unit-norm nên $R_{ij}$ chính là
  cosine giữa query $i$ và hàng index $j$; không chia chuẩn lúc truy vấn.

## Hàm nhân: `pycauset.matmul`

`A` không nằm trong RAM (xem [lưu trữ](03-retrieval.md#2-lưu-trữ-artifact-npz-đọc-qua-npy-memory-mapped)):
nó là ma trận `pycauset` file-backed. `torch.matmul` cần cả hai toán hạng nằm
trọn trong bộ nhớ, nên không dùng được. `pycauset` nhận toán hạng file-backed
và đọc payload theo tile.

```python
scores = pc.to_numpy(
    pc.matmul(pc.matrix(query_chunk), index.vectors.T),  # (b, d) @ (d, N)
    allow_huge=True,
)                                                         # -> ndarray (b, N)
```

- `index.vectors.T` là view của ma trận file-backed: chỉ đổi metadata, không
  copy payload.
- `pc.matrix(query_chunk)` copy riêng chunk query (nhỏ) vào storage của
  pycauset để phép nhân chỉ có một toán hạng file-backed.
- Kết quả là một block `(b, N)` tính trong bộ nhớ, kích thước do chunk quyết
  định (§Chunk bên dưới).

## Chunk

Toàn bộ index được quét lại ở mỗi lời gọi, nên `b` là số query của chunk chứ
không phải cả batch:

```python
chunk_rows = max(1, min(_QUERY_CHUNK, _SCORE_BUDGET // count))
```

`_QUERY_CHUNK = 256`, `_SCORE_BUDGET = 2^26` float32 (≈256 MiB): block điểm
không phình theo số hàng index lẫn kích thước batch.

Không cắt được phía index: `pycauset` từ chối matmul trên view có offset
(`matmul on views with offsets is not supported`), nên chỉ query mới chia được.

## Top-k

`pycauset` không có `topk`. Trên block `(b, N)`:

1. `np.partition` trên **bản copy** để lấy điểm lớn thứ `k` của mỗi hàng — bản
   copy giữ nguyên thứ tự cột, nên cột vẫn trỏ đúng hàng index.
2. Giữ **mọi** entry có điểm ≥ ngưỡng đó (không chỉ đúng `k`), để tie-break
   theo `(-score, cui, surface)` thay vì theo thứ tự partition.
3. Sắp xếp và cắt `k` bằng một lượt `np.lexsort` cho cả block.

## Sai số

Cả hai vế là float32; kết quả có thể lệch BLAS vài ULP ở chữ số cuối. Thứ tự
giữa các entry bằng điểm được quyết định bởi tie-break ở trên, không bởi thứ
tự mà backend trả về.
