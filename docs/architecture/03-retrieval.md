# C4 — Vector Index và truy vấn cosine

Thuộc [kiến trúc GRF](README.md). Encoder: [encoder](02-encoder.md). Fusion
nằm ở [fusion](05-fusion.md); tài liệu này chỉ mô tả **một lần quét
index** $R = Q A^\top$.

## 1. Index

`VectorIndex` giữ đúng **một entry cho mỗi surface form** của KB:

```text
IndexEntry {
  vector:   float[d]        // E(surface_form), unit-norm
  cui:      string
  surface:  string          // alias hoặc "alias (preferred name)"
  source_kb:string | null   // tag để lọc theo KB đích
}
```

Ràng buộc: **một concept có bao nhiêu alias thì có bấy nhiêu entry độc lập**;
không gộp vector theo concept ở tầng index.

Vì mọi hàng index đã unit-norm, tích vô hướng **chính là cosine** — không cần
chia chuẩn lúc truy vấn.

## 2. Lưu trữ: artifact `.npz`, đọc qua `.npy` memory-mapped

Artifact trên Drive vẫn là **một file nén** `grf_kb.npz` (`VectorIndex.save_npz`):
float32 vectors, `cuis`, `surfaces`, `source_kbs` và blob JSON `meta`; không
dùng pickle, không nhúng trọng số encoder.

Nạp index **không** đọc cả KB vào RAM:

1. `VectorIndex.load_npz(path, cache_dir=...)` stream member `vectors.npy` từ
   file `.npz` (zip) ra một thư mục cache **local** — `zipfile` + `copyfileobj`
   theo block, không `np.load` cả mảng.
2. `np.load(cache/vectors.npy, mmap_mode="r")` — chỉ mở mmap, **không** copy.
   Ma trận `pycauset` chỉ được dựng khi ai đó đọc thuộc tính `vectors` (ví dụ
   bước verify của `build_KB`); cả hai đường quét đều đọc thẳng memmap nên
   không cần nó.
3. Metadata (`cuis`, `surfaces`, `source_kbs`, `encoder_info`) đọc từ `meta`.

Cache được đặt tên theo `(kích thước, mtime)` của file `.npz`, nên build lại KB
là tự động extract lại — không bao giờ chấm điểm trên vector cũ. Mặc định cache
nằm ở thư mục tạm của máy (`tempfile.gettempdir()/grf_kb_npy`), notebook có thể
truyền `cache_dir` để đổi chỗ (ví dụ `/content/grf_kb_npy`).

> Vì sao không `np.load(mmap_mode="r")` thẳng file `.npz`: `.npz` là zip nén,
> không memory-map được. Muốn mmap phải có `.npy` thô trên đĩa — đó là lý do
> bước extract tồn tại.

## 3. Phép nhân ma trận: block hàng

Index là `.npy` mmap nên được đọc **từng block hàng** rồi nhân với cả batch
query — không dựng ma trận `(b, N)` đầy đủ:

```python
block = source[start:stop]                # memmap, blk × d
scores = queries @ block.T                # NumPy/BLAS trên CPU
# hoặc: torch.tensor(block, device="cuda") rồi queries_gpu @ a.T
cutoff = partition(scores, w - k)         # điểm lớn thứ k của mỗi hàng
rows, cols = nonzero(scores >= cutoff[:, None])
```

- Chỉ block đang tính nằm trong RAM (hoặc VRAM); KB không bao giờ nằm trọn —
  đường offload đĩa → RAM (prefetch của OS) → GPU do vòng lặp block điều phối.
- Block điểm `(q, blk)` bị chặn bởi bộ nhớ trống của thiết bị, không bởi kích
  thước index (§4).
- Không backend nào có `topk`: ngưỡng k-th lấy bằng `np.partition` /
  `torch.kthvalue`, sau đó giữ **mọi** entry ≥ ngưỡng (§4).
- Sai khác float32 giữa BLAS và torch chỉ ở ULP cuối; quy tắc thứ tự giữ nguyên.

## 4. Chunking và top-k

$$\begin{aligned}
S_{\text{block}} &= Q \cdot A_{\text{block}}^\top \in \mathbb{R}^{q \times b} \\
\text{top-}k(S_{\text{block}}[i, :]) &\to \{(a, s_{ia})\}_{a \in \text{Top-}k}
\end{aligned}$$

- Số hàng index mỗi block = `free_bytes // 3 // (4·d + 5·q)`: một hàng tốn
  `4·d` byte vector + `4+1` byte mỗi query (điểm + mask), dùng một phần ba bộ
  nhớ trống (`MemAvailable` trên CPU, `torch.cuda.mem_get_info` trên GPU).
- Top-k dùng partition trên **bản copy** của block: giữ nguyên thứ tự cột,
  nên cột vẫn trỏ đúng hàng index; mọi entry bằng điểm cắt được giữ, để tie-break
  theo `(-score, cui, surface)` thay vì theo thứ tự partition.
- Lọc `source_kb`: chỉ chọn **cột** của cùng block điểm đó, nên một hàng được
  chấm điểm y hệt dù có lọc hay không.

## 5. GPU

`index.use_gpu("cuda")` chấm điểm bằng **torch** (đã có sẵn cho encoder). `pycauset` là bản CPU, không
có backend CUDA. Index vẫn là `.npy` memmap trên đĩa: từng **block hàng** được
copy sang VRAM, `Q · A_blockᵀ` chạy trên GPU, chỉ các entry ≥ ngưỡng k-th của
block trả về CPU. KB không bao giờ nằm trọn trong VRAM; cỡ block lấy từ VRAM
trống (`cp.cuda.Device().mem_info`).

Notebook bọc lời gọi trong `try/except RuntimeError` và rơi về CPU nếu môi
trường không có CUDA; `index.search_device` cho biết đang chạy ở đâu.
Chi tiết phép nhân: [matmul](08-matmul.md).

## 6. Lô hoá

```text
A ∈ R^{n_surface × d}      # ma trận index, mỗi hàng một surface form
Q ∈ R^{b × d}              # b query của một batch
R  = Q · A^T               # (b × n_surface) cosine, vì mọi hàng đã unit-norm
top-k theo từng hàng của R → candidate của mention đó
```

Pipeline 1 gọi phép này **một lần** với $Q = Q_{\text{GRF}}$ của cả batch.
Pipeline 2 gọi nó **n+1 lần**, lần thứ $x$ với $Q = Q_x$ ([fusion §7.3](05-fusion.md#73-pha-2b-rrf-retrieval-post-retrieval)),
nên mỗi lần vẫn là một phép ma trận duy nhất cho cả batch chứ không phải một
vòng lặp mention.

Baseline encode hết query rồi chấm điểm một lần cho cả tập; GRF encode toàn bộ
`Q_0` trong `link_batch` bằng một lượt forward, sau đó sinh feedback và chấm
điểm cho từng request.

## 7. Concept aggregation (vectorized)

$$\begin{aligned}
\text{Score}(c) &= \max_{a \in \mathcal{A}_c \cap \text{Top-}k} S(a) \\
\text{Surface}_{\text{rep}}(c) &= \arg\max_{a \in \mathcal{A}_c \cap \text{Top-}k} \big( S(a), -\text{lex}(a) \big)
\end{aligned}$$

Triển khai: một lượt `np.lexsort` trên khoá $(-\text{score}, \text{cui},
\text{surface})$ cho toàn bộ block $(m \times k)$, rồi cắt `limit` phần tử đầu
của từng hàng bằng `np.bincount` + `cumsum` (không có vòng lặp Python per
mention). Tie-break surface: lex ascending.

## 8. Thứ tự cuối và top-k concept

$$ \text{Candidates} = \text{argsort}_{c} \big( -\text{Score}(c), c \big)[:k_{\text{concept}}] $$

Trả về danh sách `(cui, score, surface_rep)` đã sắp xếp; đây là đầu vào của
re-ranker (stage riêng biệt, không thuộc retrieval).
