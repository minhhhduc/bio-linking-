# C6 — Fusion

Thuộc [kiến trúc GRF](README.md). Index và truy vấn cosine: [retrieval](03-retrieval.md).

`feedback_types=()` ⇒ $Q_{\text{GRF}} = Q_0$ và hai pipeline trùng nhau (baseline).
Đây là lý do cùng một mã nguồn chạy được cả ba Bảng 1–3.

## 5.2. Pipeline 1 — Vector-based GRF (pre-retrieval)

```text
m, s ──▶ E_ctx(m, s) = Q_0           // pool token của mention trong câu
     ──▶ LLM(m, s, t_i) = F_i        (song song cho từng t_i)
     ──▶ E(F_i) = Q_Fi
     ──▶ Q_GRF = w_0·Q_0 + Σ w_i·Q_Fi      (w_0 = w_i = 1/(n+1), §7.2)
     ──▶ cosine(Q_GRF, I) → top-k surface forms
     ──▶ score(c) = max_{a ∈ A_c ∩ top-k} S(a)
     ──▶ top-k concept → re-ranker / pick top-1
```

Đặc điểm: **chỉ một lần quét index** — chi phí retrieval không tăng theo số loại
feedback, chỉ tăng ở bước encode $F_i$.

Cấu hình trọng số:

- Đơn lẻ ($n = 1$): $w_0 = w_1 = \frac{1}{n+1} = 0.5$.
- All-in-One ($n$ loại): mỗi thành phần trọng số $\frac{1}{n+1}$.

Toàn bộ batch được xử lý theo các pha ma trận độc lập ở §7, không có vòng lặp
Python cho từng mention.

## 5.3. Pipeline 2 — Rank-based GRF (post-retrieval)

```text
m, s ──▶ E_ctx(m, s) = Q_0     ──▶ Search(Q_0, I)  = R_0
     ──▶ LLM(m, s, t_i) = F_i  ──▶ E(F_i) = Q_Fi   ──▶ Search(Q_Fi, I) = R_Fi
     ──▶ RRF(c) = Σ w_x · 1/r_x(c)        (w_x = 1/(n+1))
     ──▶ sort desc → top-k concept → re-ranker
```

Đặc điểm: **n+1 lần quét index, lần lượt theo từng component** (mỗi lần là một
ma trận $Q_x$ của cả batch, §7.3), bù lại dung hợp theo thứ hạng bền vững hơn
với khác biệt thang điểm giữa các query.

## 5.4. Chọn pipeline

| Tiêu chí | Vector-based | Rank-based |
| --- | --- | --- |
| Số lần truy vấn index | 1 | $n+1$ (mỗi lần một ma trận $Q_x$) |
| Chi phí encode feedback | có | có |
| Độ bền với lệch thang điểm | trung bình | cao |
| Độ trễ | thấp | cao hơn |
| Khi nào dùng | index lớn, cần nhanh | chất lượng ưu tiên, top-k lớn |

## 7.1. Pha 1 — Encode queries (batch)

$$
\begin{aligned}
Q_0 &\in \mathbb{R}^{b \times d} && \text{(} b \text{ mention, một lượt forward)} \\
F &\in \mathbb{R}^{b \times n \times \ell} && \text{(feedback texts, song song per request)} \\
Q_F &\in \mathbb{R}^{b \times n \times d} && \text{(encode toàn bộ feedback, một lượt)}
\end{aligned}
$$

- $b$: batch size (số mention)
- $n$: số feedback component thành công per request
- $d$: encoder dimension
- $\ell$: độ dài text (biến thiên)

## 7.2. Pha 2a — Vector interpolation (pre-retrieval)

$$
\begin{aligned}
w_0 &= \frac{1}{n+1}, \quad w_i = \frac{1}{n+1} \quad \forall i \in [1, n] \\
Q_{\text{GRF}}[j] &= w_0 \cdot Q_0[j] + \sum_{i=1}^{n_j} w_i \cdot Q_F[j, i] && \text{per mention } j \\
Q_{\text{GRF}} &= \text{L2-normalize}(Q_{\text{GRF}}) \in \mathbb{R}^{b \times d}
\end{aligned}
$$

Đây là **phép chuẩn hoá duy nhất** của một vector retrieval (xem
[encoder](02-encoder.md#quy-tắc-chuẩn-hoá-l2)).

## 7.3. Pha 2b — RRF retrieval (post-retrieval)

$n$ feedback ⇒ $n+1$ vector embedding **rời nhau**, tính cả mention gốc:

$$
\vec{Q}_0 = E_{\text{ctx}}(m, s), \quad
\vec{Q}_{F_1}, \ldots, \vec{Q}_{F_n}
$$

Batch hoá theo **component**, không theo mention: mỗi component là một ma trận
$Q_x$ riêng, hàng là các mention của batch. Với batch $b$ mention, mỗi mention
$n$ feedback (component thứ $i$ của mention $j$ là $\vec{Q}_i[j]$):

$$
\begin{aligned}
Q_0 &= \begin{bmatrix} \vec{Q}_0[1] \\ \vdots \\ \vec{Q}_0[b] \end{bmatrix}
\in \mathbb{R}^{b \times d},
&
Q_1 &= \begin{bmatrix} \vec{Q}_1[1] \\ \vdots \\ \vec{Q}_1[b] \end{bmatrix}
\in \mathbb{R}^{b \times d},
&
\ldots,
&
Q_n &= \begin{bmatrix} \vec{Q}_n[1] \\ \vdots \\ \vec{Q}_n[b] \end{bmatrix}
\in \mathbb{R}^{b \times d}
\end{aligned}
$$

**Retrieval lần lượt**: quét index $n+1$ lần, lần lượt theo component, mỗi lần
dùng đúng công thức retrieval batch ([retrieval §6](03-retrieval.md#6-lô-hoá))
với $Q_x$ — thu về $n+1$ ma trận thứ hạng, mỗi ma trận $b$ hàng:

$$
R_x = \text{CosineTopK}(Q_x, A) \in \mathbb{R}^{b \times k},
\quad x = 0, \ldots, n
$$

Mỗi hàng $j$ của $R_x$ là danh sách hit của $\vec{Q}_x[j]$; sau khâu aggregation
mỗi hàng trở thành rank list $R_x[j] = \{c \mapsto r_x(c)\}$.

Feedback lỗi ⇒ mention đó vắng mặt ở batch về sau, không pad giả: mention có
$n_j$ feedback thì đóng góp đúng $n_j+1$ rank list, trọng số $1/(n_j+1)$.

## 7.6. Pha 5 — RRF fusion (nếu dùng pipeline 2)

Với mention $j$, vector thứ hạng của khái niệm $c$ là
$\big( r_0(c), r_1(c), \ldots, r_{n_j}(c) \big)$ — đúng một thành phần cho mỗi
lần retrieval ở §7.3:

$$
\text{RRF}(c) = \sum_{x=0}^{n_j} w_x \cdot \frac{1}{r_x(c)},
\qquad w_x = \frac{1}{n_j+1},
\qquad \frac{1}{r_x(c)} = 0 \text{ nếu } c \notin R_x
$$

Dùng đúng ma trận thứ hạng của §7.3: dựng mảng phẳng `owner * n_concepts + cui_id`
và cộng $w_x / r_x(c)$ bằng `np.bincount`, thu về ma trận điểm $(b \times |\mathcal{C}|)$
— một scatter-add cho cả $n+1$ rank list, không phải một dict mỗi query.
