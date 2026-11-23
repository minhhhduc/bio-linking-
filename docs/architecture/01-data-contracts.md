# Dữ liệu và hợp đồng giữa các tầng

Thuộc [kiến trúc GRF](README.md).

## 4.1. Bản ghi KB

KB do người dùng cung cấp; kiến trúc không ràng buộc nguồn KB cụ thể.

```text
Concept {
  cui:      string          // mã concept do KB cung cấp
  name:     string          // N_c — preferred name
  aliases:  string[]        // A_c — surface forms
}
```

## 4.2. Bản ghi index

```text
IndexEntry {
  vector:   float[d]        // E(surface_form)
  cui:      string
  surface:  string          // alias hoặc "alias (preferred name)"
}
```

Ràng buộc: **một concept có bao nhiêu alias thì có bấy nhiêu entry độc lập**; không
gộp vector theo concept ở tầng index.

## 4.3. Đầu vào / đầu ra pipeline

```text
Request  { mention: string, context: string, feedback_types: string[], n_syn: int,
           start_token?: int, end_token?: int }   // span token của mention trong context
Response { cui: string, candidates: [{cui, score, surface}], debug: {Q_0, Q_F, ranks} }
```

Bản gold tiếng Anh (cho thí nghiệm song ngữ) không do pipeline này sinh ra: nó
được dịch offline từ gold tiếng Việt, giữ nguyên `cui` / `icd10` / `label` và mọi
field `vi_*`, chỉ thêm `en_sentence_str`, `en_term` và span `en_*` tương ứng —
xem [`../translate-gold-pipeline.md`](../translate-gold-pipeline.md).

## 4.4. Nguyên tắc chung: mention luôn nằm trong ngữ cảnh

Mỗi mention $m$ luôn đi kèm câu chứa nó $s$ (`Request.context`); không có chế độ
"mention rời". Dense query dùng chung cho **baseline và GRF** giữ nguyên câu và
đánh dấu mention **tại đúng vị trí của nó**:

```text
text = s[:start] + " [SEP] " + m + " [SEP] " + s[end:]
Q_0  = mean-pool( hidden states của các token mention nằm giữa hai [SEP] )
```

- `start` / `end` là span ký tự của mention trong câu, suy ra từ span token gold
  (`start_token`, `end_token`, token hoá bằng khoảng trắng); nếu thiếu thì định
  vị mention bằng so khớp chuỗi.
- Phần câu **trước** và **sau** mention vẫn là context: tham gia self-attention
  nên vector của mention phụ thuộc ngữ cảnh, nhưng **không** được mean vào
  vector — tránh pha loãng tín hiệu mention.
- `[CLS]` / `[SEP]` (offset 0) bị loại; chỉ token thực giao với span ký tự của
  mention được đưa vào phép mean.
- Truncation câu dài có thể cắt mất mention: encoder báo lỗi thay vì lặng lẽ
  pool nhầm token.
- $Q_0$ là **duy nhất** phép mean-pool đó: không cộng thêm $E(m)$, không thêm
  thành phần nào khác.

Phía index KB vẫn encode surface form độc lập ($E(a)$), vì tên khái niệm không
có ngữ cảnh. Baseline và GRF dùng cùng một hàm
(`SentenceTransformerEncoder.encode_mention_in_context`) cho $Q_0$, nên hai bên
retrieve trên cùng một biểu diễn và số liệu so sánh được.

**Mỗi vector retrieval chỉ được L2-normalize đúng một lần, sau khi $Q_{\text{GRF}}$
đã hình thành** ([fusion §7.2](05-fusion.md#72-pha-2a-vector-interpolation-pre-retrieval)):
$Q_0$ là mean-pool **thô**, không chia chuẩn; mỗi $Q_{F_i} = \frac{1}{n}\sum_j E(F_j)$
cũng thô; phép chia chuẩn duy nhất nằm ở bước nội suy, sau phép cộng có trọng số.
Chuẩn hoá $Q_0$ (hay bất kỳ $Q_{F_i}$) trước khi cộng sẽ vừa là lần chuẩn hoá thứ
hai của cùng vector, vừa gán cho các thành phần một trọng số 1:1 cố định — không
phải công thức. Khi `feedback_types=()`, $Q_{\text{GRF}} = Q_0$ nên phép chuẩn hoá
duy nhất đó vẫn nằm ở fusion (baseline: ở `VectorIndex.search_batch`, nơi nhận
đúng một lần chuẩn hoá cho query thô).

Hai notebook `ranking` (baseline và GRF) có công tắc `USE_CONTEXT_QUERY`: `True` lấy
$Q_0 = \text{mean-pool}(\text{token mention trong câu})$ như trên; `False` bỏ ngữ
cảnh và lấy $Q_0 = E(m)$ bằng `encoder.encode_raw` — đúng hàm đã dựng index row,
trước khi chuẩn hoá.

## 4.5. Lưu trữ

Bố cục thư mục (Drive, candidate, feedback bundle, cache `.npy`) xem
[`../storage-layout.md`](../storage-layout.md).
