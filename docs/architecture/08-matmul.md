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

## Hàm nhân

`A` là ma trận `pycauset` file-backed (xem
[lưu trữ](03-retrieval.md#2-lưu-trữ-artifact-npz-đọc-qua-npy-memory-mapped)):
payload không nằm trong RAM. Có hai đường:

**CPU — `pycauset.matmul`.** `pycauset` nhận toán hạng file-backed và đọc
payload theo tile:

```python
scores = pc.to_numpy(
    pc.matmul(pc.matrix(query_chunk), index.vectors.T),  # (b, d) @ (d, N)
    allow_huge=True,
)                                                         # -> ndarray (b, N)
```

**GPU — CuPy trên từng block hàng.** `pycauset` không có backend CUDA, nên
`use_gpu()` đọc `.npy` bằng memmap và đẩy từng block sang VRAM:

```python
a = cp.asarray(source[start:stop])        # (blk, d), chỉ một block
scores = queries @ a.T                    # chạy trên GPU
cutoff = cp.partition(scores, w - k, axis=1)[:, w - k]
rows, cols = cp.nonzero(scores >= cutoff[:, None])
```

Cỡ block lấy từ VRAM trống (`cp.cuda.Device().mem_info`): một hàng index tốn
`4·d` byte vector + `4+1` byte mỗi query (điểm + mask), dùng một phần ba
VRAM trống. KB không bao giờ nằm trọn trong VRAM; kết quả chỉ là các entry
≥ ngưỡng k-th của block.

- `index.vectors.T` (CPU) là view của ma trận file-backed: chỉ đổi metadata,
  không copy payload.
- Kết quả mỗi block là một ma trận `(b, w)` tính trong bộ nhớ, kích thước do
  block quyết định.

## Chunk (đường CPU)

Toàn bộ index được quét lại ở mỗi lời gọi, nên `b` là số query của chunk chứ
không phải cả batch:

```python
chunk_rows = max(1, min(_QUERY_CHUNK, _SCORE_BUDGET // count))
```

`_QUERY_CHUNK = 256`, `_SCORE_BUDGET = 2^26` float32 (≈256 MiB): block điểm
không phình theo số hàng index lẫn kích thước batch.

Không cắt được phía index trên đường CPU: `pycauset` từ chối matmul trên view
có offset (`matmul on views with offsets is not supported`), nên chỉ query mới
chia được. Đường GPU thì ngược lại — index được cắt thành block hàng, query
giữ nguyên cả batch.

## Top-k

Backend không có top-k. Trên mỗi block `(b, w)`:

1. Partition (`np.partition` trên CPU, `cp.partition` trên GPU) trên **bản
   copy** để lấy điểm lớn thứ `k` của mỗi hàng — bản copy giữ nguyên thứ tự
   cột, nên cột vẫn trỏ đúng hàng index.
2. Giữ **mọi** entry có điểm ≥ ngưỡng đó (không chỉ đúng `k`), để tie-break
   theo `(-score, cui, surface)` thay vì theo thứ tự partition.
3. Sắp xếp và cắt `k` bằng một lượt `np.lexsort` cho cả block (sau khi các
   block đã gộp — một block chỉ có thể góp tối đa `k` entry vào top-k toàn cục,
   nên cắt theo ngưỡng riêng của block không làm mất ứng viên).

## Sai số

Cả hai vế là float32; kết quả có thể lệch BLAS vài ULP ở chữ số cuối. Thứ tự
giữa các entry bằng điểm được quyết định bởi tie-break ở trên, không bởi thứ
tự mà backend trả về.
