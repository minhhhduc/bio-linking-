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

`A` không nằm trong RAM: nó là file `.npy` mmap (xem
[lưu trữ](03-retrieval.md#2-lưu-trữ-artifact-npz-đọc-qua-npy-memory-mapped)).
Cả hai đường đều đọc **từng block hàng** rồi nhân với cả batch query:

**CPU — NumPy/BLAS.**

```python
block = source[start:stop]                # memmap view, blk × d
scores = queries @ block.T                # (q, d) @ (d, blk)
cutoff = np.partition(scores, w - k, axis=1)[:, w - k]
rows, cols = np.nonzero(scores >= cutoff[:, None])
```

**GPU — torch.**

```python
a = torch.tensor(source[start:stop], device="cuda")
scores = queries_gpu @ a.T
cutoff = torch.kthvalue(scores, w - k + 1, dim=1).values
rows, cols = (scores >= cutoff[:, None]).nonzero(as_tuple=True)
```

Cỡ block lấy từ bộ nhớ trống mà thiết bị báo (`MemAvailable` trên CPU,
`torch.cuda.mem_get_info` trên GPU): một hàng index tốn `4·d` byte vector +
`4+1` byte mỗi query (điểm + mask), dùng một phần ba. KB không bao giờ nằm
trọn trong RAM/VRAM; kết quả mỗi block chỉ là các entry ≥ ngưỡng k-th.

`pycauset` chỉ còn dùng khi ai đó đọc thuộc tính `vectors` (ví dụ bước verify
của `build_KB`), và lúc đó ma trận mới được dựng — không đường quét nào phụ
thuộc vào nó.

## Chunk

Vì mỗi lượt gọi quét toàn index, đường cũ chia **query** thành chunk
(`_QUERY_CHUNK = 256`, `_SCORE_BUDGET = 2^26` float32 ≈256 MiB) để block điểm
không phình. Đường hiện tại chia **index** thành block hàng: query giữ nguyên
cả batch, block điểm `(q, blk)` bị chặn bởi bộ nhớ thiết bị. Không cần cắt
phía index bằng view có offset — block được đọc trực tiếp từ memmap.

## Top-k

Trên mỗi block `(b, w)`:

1. Partition (`np.partition` trên CPU, `torch.kthvalue` trên GPU) trên **bản
   copy** để lấy điểm lớn thứ `k` của mỗi hàng — bản copy giữ nguyên thứ tự
   cột, nên cột vẫn trỏ đúng hàng index.
2. Giữ **mọi** entry có điểm ≥ ngưỡng đó (không chỉ đúng `k`), để tie-break
   theo `(-score, cui, surface)` thay vì theo thứ tự partition.
3. Sắp xếp và cắt `k` bằng một lượt `np.lexsort` cho cả block — một block chỉ
   có thể góp tối đa `k` entry vào top-k toàn cục, nên cắt theo ngưỡng riêng
   của block không làm mất ứng viên.

## Sai số

Cả hai vế là float32; kết quả có thể lệch BLAS vài ULP ở chữ số cuối. Thứ tự
giữa các entry bằng điểm được quyết định bởi tie-break ở trên, không bởi thứ
tự mà backend trả về.
