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
2. `np.load(cache/vectors.npy, mmap_mode="r")` rồi
   `pycauset.matrix(..., max_in_ram_bytes=512 MiB)`: pycauset ghi payload vào
   một **backing file** trên đĩa và ma trận chỉ trỏ tới file đó, nên không cấp
   phát $N \times d$ float trong heap. `np.load` thường (hay `pycauset.load_npy`,
   vốn `np.load` rồi copy) nạp cả mảng vào RAM và bị `MemoryGovernor` từ chối
   khi KB lớn hơn RAM trống.
3. Metadata (`cuis`, `surfaces`, `source_kbs`, `encoder_info`) đọc từ `meta`.

Cache được đặt tên theo `(kích thước, mtime)` của file `.npz`, nên build lại KB
là tự động extract lại — không bao giờ chấm điểm trên vector cũ. Mặc định cache
nằm ở thư mục tạm của máy (`tempfile.gettempdir()/grf_kb_npy`), notebook có thể
truyền `cache_dir` để đổi chỗ (ví dụ `/content/grf_kb_npy`).

> Vì sao không `np.load(mmap_mode="r")` thẳng file `.npz`: `.npz` là zip nén,
> không memory-map được. Muốn mmap phải có `.npy` thô trên đĩa — đó là lý do
> bước extract tồn tại.

## 3. Phép nhân ma trận: `pycauset.matmul`

Retrieval **không dùng `torch.matmul`**. Phép chấm điểm là một lời gọi
`pycauset.matmul` trên operand file-backed:

```python
scores = pc.to_numpy(
    pc.matmul(pc.matrix(query_chunk), index.vectors.T),  # (b, d) @ (d, N)
    allow_huge=True,
)                                                         # -> ndarray (b, N)
```

- `index.vectors.T` là **view** của ma trận file-backed (chỉ đổi metadata,
  không copy payload).
- Operand file-backed được đọc theo **tile**: chỉ tile đang tính nằm trong
  RAM/VRAM, không bao giờ cả ma trận — đường offload đĩa → RAM (prefetch) →
  GPU do `pycauset` tự điều phối.
- Kết quả `(b, N)` là ma trận tính toán trong RAM, kích thước bị chặn bởi
  chunk (xem §4); `np.asarray`/`to_numpy` trên block nhỏ là hợp lệ.
- `pycauset` không có `topk`: top-k làm bằng `numpy` trên block điểm đó (§4).
- Sai khác float32 so với BLAS chỉ ở ULP cuối; quy tắc thứ tự giữ nguyên.

## 4. Chunking và top-k

$$\begin{aligned}
S_{\text{chunk}} &= Q_{\text{chunk}} \cdot A^\top \in \mathbb{R}^{c \times N} \\
\text{top-}k(S_{\text{chunk}}[i, :]) &\to \{(a, s_{ia})\}_{a \in \text{Top-}k}
\end{aligned}$$

- Kích thước block điểm: `_score_block_bytes()` = $\lceil$`MemAvailable`$/6\rceil$,
  kẹp trong 256 MiB..8 GiB; số query mỗi chunk = `block_bytes // 4 // N`. Mỗi
  chunk là **một lượt quét toàn index**, nên block càng rộng càng ít lượt:
  959 query trên index 3M hàng, 256 MiB cần ~11 lượt, 4 GiB chỉ ~2.
- Trong lúc quét, ngưỡng RAM của `pycauset` được nâng lên đúng cỡ block rồi trả
  lại trong `finally`: block nằm trong RAM thay vì spill ra backing file rồi
  đọc lại để top-k.
- Không chia phía index được: `pycauset` từ chối matmul trên view có offset
  (`matmul on views with offsets is not supported`), nên chỉ query mới cắt được.
- Top-k dùng `np.partition` trên **bản copy** của block: giữ nguyên thứ tự cột,
  nên cột vẫn trỏ đúng hàng index; mọi entry bằng điểm cắt được giữ, để tie-break
  theo `(-score, cui, surface)` thay vì theo thứ tự partition.
- Lọc `source_kb`: chỉ chọn **cột** của cùng block điểm đó, nên một hàng được
  chấm điểm y hệt dù có lọc hay không.

## 5. GPU

`index.use_gpu("cuda")` gọi `pycauset.cuda.enable()`; `pycauset` tự route phép
nhân sang thiết bị. Không còn bản copy `torch` của toàn bộ KB trong VRAM/RAM —
ma trận vẫn file-backed, chỉ chunk truy vấn và block điểm đi qua thiết bị.

Notebook bọc lời gọi trong `try/except RuntimeError` và rơi về CPU nếu bản
`pycauset` không có CUDA; `index.search_device` cho biết đang chạy ở đâu.

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
