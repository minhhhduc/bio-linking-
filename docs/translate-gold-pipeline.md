# Pipeline dịch gold data (VI → EN)

Tài liệu mô tả notebook [`source/notebooks/common/translate_gold.ipynb`](../source/notebooks/common/translate_gold.ipynb):
dịch `vi_sentence_str` và `vi_term` của gold BEL JSONL sang tiếng Anh bằng một
checkpoint Qwen cục bộ, giữ nguyên toàn bộ metadata gốc và sinh lại token span
cho phía tiếng Anh.

Mục tiêu không chỉ là "dịch xong", mà là **span tiếng Anh phải đúng**: mỗi
`en_term` phải nằm đúng vị trí của nó trong `en_sentence_str`, kể cả khi term
lặp nhiều lần trong cùng câu hoặc khi model diễn đạt lại (paraphrase).

## 1. Dữ liệu vào / ra

| Vai trò | Đường dẫn (trong `DRIVE_DATA_DIR`) |
| --- | --- |
| Vào | `test_fixed_verified_gold_revised.jsonl` |
| Ra | `test_fixed_verified_gold_revised_en.jsonl` |
| Cache | `vi_en_translation_cache.jsonl` (append-only) |

Bản local tương ứng phần vào: [`data/test_fixed_verified_gold_revised.jsonl`](../data/test_fixed_verified_gold_revised.jsonl).

Record vào:

```json
{"entities": [{"cui": "C0023343", "icd10": ["A30.9", "A30"], "label": "DISEASE",
               "vi_term": "bệnh phong",
               "vi_start_token_idx": 8, "vi_end_token_idx": 9}],
 "vi_sentence_str": "dạng phối hợp : ..."}
```

Record ra giữ nguyên mọi field trên, chỉ thêm:

```json
{"entities": [{"cui": "C0023343", "icd10": ["A30.9", "A30"], "label": "DISEASE",
               "vi_term": "bệnh phong",
               "vi_start_token_idx": 8, "vi_end_token_idx": 9,
               "en_term": "leprosy",
               "en_start_token_idx": 8, "en_end_token_idx": 8}],
 "vi_sentence_str": "dạng phối hợp : ...",
 "en_sentence_str": "combined form : ... leprosy ..."}
```

`cui`, `icd10`, `label` và toàn bộ field `vi_*` là **read-only**: pipeline không
sửa, không chuẩn hoá, không đổi thứ tự entity.

## 2. Sơ đồ tổng thể

```text
  gold JSONL (VI)
        │
        ▼
  ┌───────────────────────────────────────────────────────────────┐
  │ Stage 1-2  mount Drive, validate từng dòng                     │
  │            vi_sentence_str + entities + vi_term phải có thật   │
  └───────────────────────────────────────────────────────────────┘
        │ 1476 record / 3686 entity
        ▼
  ┌───────────────────────────────────────────────────────────────┐
  │ Stage 3  tách 2 luồng công việc                                │
  │                                                                │
  │   glossary:  vi_term unique        → 1712 chuỗi                │
  │   câu mask:  mention → [[Tn]]      → 1470 chuỗi                │
  │              (mask giống nhau thì dedup)                       │
  │                                                                │
  │   pending = chuỗi chưa có trong cache        → 3182 chuỗi      │
  └───────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌───────────────────────────────────────────────────────────────┐
  │ Stage 4  load model 4-bit NF4 (auto device_map)                │
  └───────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌───────────────────────────────────────────────────────────────┐
  │ Stage 5  generate greedy, batch 8                              │
  │                                                                │
  │   term  → prompt dịch term thuần                               │
  │   câu   → prompt dịch + giữ nguyên placeholder                 │
  │                                                                │
  │   ├─ hợp lệ        → ghi cache + flush từng string             │
  │   ├─ placeholder rơi → KHÔNG cache → dịch lại câu gốc (repair) │
  │   └─ answer rỗng   → KHÔNG cache → retry lần chạy sau          │
  └───────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌───────────────────────────────────────────────────────────────┐
  │ Stage 6  ghép record                                           │
  │                                                                │
  │   có marker  → thay [[Tn]] bằng en_term, span = vị trí chèn    │
  │   không có   → search en_term trong câu dịch, lấy occurrence 1 │
  │   không thấy → span null + đếm + báo record/entity             │
  └───────────────────────────────────────────────────────────────┘
        │
        ▼
  ┌───────────────────────────────────────────────────────────────┐
  │ Stage 7  verify file đã ghi, span bọc đúng en_term, giải phóng │
  └───────────────────────────────────────────────────────────────┘
        │
        ▼
  gold JSONL (VI + EN)
```

## 3. Neo span bằng placeholder (Stage 3, 5, 6)

Đây là phần cốt lõi, thay cho cách cũ (dịch câu và term rời nhau rồi đi tìm
term trong câu dịch — sai khi term lặp hoặc bị paraphrase).

### 3.1. Mask câu nguồn (Stage 3)

1. Chuẩn hoá câu: NFC + gộp whitespace, rồi `split()` — cùng quy ước token với
   [`source/common/data.py`](../source/common/data.py).
2. Với từng entity, lấy span từ `vi_start_token_idx` / `vi_end_token_idx`.
   Span chỉ được dùng khi: hai index là số nguyên hợp lệ, `start <= end`, nằm
   trong câu, và text của span khớp `vi_term`.
3. Nếu span không dùng được: lấy occurrence đầu tiên của `vi_term` chưa bị
   mention khác chiếm. Không tìm được → `unlocalised += 1`, entity đó không có
   marker.
4. Các span không bao giờ chồng lấn (mention nào chiếm token thì token đó bị
   đánh dấu), nên câu mask là duy nhất và không nhập nhằng.
5. Thay span thứ $k$ (theo thứ tự trong câu) bằng `[[Tk]]`. Đánh số theo thứ tự
   text làm mask có tính chuẩn tắc, nhờ đó hai record cùng câu cùng bộ mention
   dùng chung một job dịch.

Ví dụ:

```text
gốc   : dạng phối hợp : những người mắc loại bệnh phong này ... phong thể củ ... phong thể u .
mask  : dạng phối hợp : những người mắc loại [[T0]] này ... [[T1]] ... [[T2]] .
```

Term lặp trong cùng câu nhận marker khác nhau, nên không còn cảnh 294 mention bị
gán chung một span như cách cũ.

Ngoại lệ: nếu câu chỉ còn toàn marker (không còn chữ nào để dịch) và có nhiều hơn
một entity, pipeline lùi về câu không mask cho record đó — mất span EN của record
ấy, và được đếm vào `unlocalised`.

### 3.2. Dịch (Stage 5)

| | term | câu có marker |
| --- | --- | --- |
| System prompt | dịch term, giữ gene/drug/abbreviation/số | prompt term + "copy mọi placeholder y nguyên, đúng thứ tự, đúng chỗ" |
| Sinh | greedy (`do_sample=False`) | greedy |
| Hậu xử lý | bỏ block `thinking`, bỏ quote, gộp về một dòng | như term, thêm kiểm tra placeholder |

Kiểm tra placeholder: danh sách marker trong kết quả phải **bằng đúng** danh sách
trong input (cùng số, cùng thứ tự). Sai → kết quả bị loại, không ghi cache.

Tham số sinh: `MAX_INPUT_TOKENS = 512`, `MAX_NEW_TOKENS = 256`,
`BATCH_SIZE = 8`, `enable_thinking = False` (Qwen3 mặc định suy luận; bật
thinking thì 256 token bị ăn hết và trả về rỗng).

### 3.3. Thay marker và suy ra span (Stage 6)

Với mỗi record, marker được tách khỏi từ dính bằng regex (chèn khoảng trắng quanh
`[[Tn]]`) để nó token hoá độc lập, rồi duyệt từng token của bản dịch:

- token thường → append;
- token là `[[Tk]]` → append các token của `en_term` tương ứng và ghi
  `(start, end) = (len(tokens) trước khi chèn, len(tokens) sau khi chèn - 1)`.

Vì term được **chèn** chứ không phải **tìm**, `en_start_token_idx` /
`en_end_token_idx` đúng theo cấu trúc, không phụ thuộc việc model có lặp lại
term hay không.

## 4. Nhánh lỗi và fallback

| Tình huống | Xử lý | Hệ quả |
| --- | --- | --- |
| Model làm rơi / đổi thứ tự / nhân đôi placeholder | không cache; dịch lại **câu gốc không mask** và lưu bản dịch đó | span tìm bằng first-occurrence; nếu không thấy → `null` |
| Answer rỗng | không cache, đếm vào `failures` | lần chạy sau tự retry |
| OOM | `empty_cache()`, retry từng string một | batch nhỏ hơn, chậm hơn |
| Term không neo được vào câu nguồn | không tạo marker | entity đó rơi vào nhánh search |
| Không tìm thấy term trong câu dịch | `en_start_token_idx = en_end_token_idx = null`, đếm vào `unmatched`, log record/entity/term | Stage 7 cảnh báo, không im lặng |
| Thiếu bản dịch của bất kỳ chuỗi nào | Stage 6 `raise` trước khi ghi file | không sinh ra file nửa vời |

Nguyên tắc: **không bao giờ** ghi một span mà mình không kiểm chứng được.

## 5. Cache và resume

- File cache JSONL append-only: `{"source": ..., "target": ...}`, ghi và `flush`
  ngay sau từng string → chết runtime mất tối đa một batch.
- Khoá cache là chuỗi nguồn đã chuẩn hoá. Term và câu mask dùng chung một file
  nhưng không đụng khoá nhau (câu mask chứa `[[Tn]]`).
- Chạy lại: record nào đã có target thì bỏ khỏi `pending`; nhánh repair cũng đọc
  lại bản dịch câu gốc từ cache nên không tốn thêm lượt gọi.
- Cache **không** gắn với model/prompt: đổi checkpoint mà giữ cache thì bản dịch
  cũ vẫn được tái sử dụng. Muốn dịch lại từ đầu thì xoá file cache.

## 6. Kiểm chứng

Stage 7 đọc lại file vừa ghi và kiểm tra những gì consumer thực sự dựa vào:

- số record khớp lần ghi;
- mọi record có `en_sentence_str`;
- mọi entity có `en_term`;
- mọi span trong `[0, len(tokens))` và `start <= end`;
- `" ".join(tokens[start:end+1]) == en_term`.

Kết quả dry-run trên gold data hiện tại (bản dịch giả lập, không cần GPU):

```text
Unique strings: 1,470 sentences + 1,712 terms = 3,182
Spans: 3,685 from placeholders, 1 searched, 1 null
Verified: 1,476 records, 0 field VI/CUI/ICD bị thay đổi
```

Một span `null` duy nhất: record 372, gold ghi câu là `amitripxylin` còn
`vi_term` là `amitriptylin` (hai cách gõ khác nhau của *amitriptyline*), nên term
không tồn tại nguyên văn trong câu — lỗi data, không phải lỗi pipeline.

## 7. Cấu hình và chạy

Bắt buộc sửa trong cell *Constants and Configuration*:

| Hằng số | Ghi chú |
| --- | --- |
| `DRIVE_DATA_DIR` | folder đã tồn tại trên Drive; notebook **không** tạo folder |
| `MODEL_ID` | checkpoint Qwen thật (Hub id hoặc thư mục local). Id mẫu trong notebook **không** tồn tại trên Hub |
| `LOAD_IN_4BIT` | 4-bit NF4 + double quant, ~15 GB cho model 27B; tắt nếu GPU đủ lớn |
| `LIMIT` | `None` = dịch hết; đặt int để smoke test |

Chạy tuần tự Stage 1 → 7. Lần đầu Stage 4 sẽ tải weight; nếu runtime bị ngắt, chạy
lại từ Stage 1 vẫn tiếp tục nhờ cache.

## 8. Hạn chế đã biết

- Chất lượng phụ thuộc checkpoint và việc model tôn trọng placeholder. Placeholder
  hỏng không làm hỏng dataset (có fallback), nhưng làm giảm số span chính xác.
- Bản dịch term và bản dịch câu là hai lần gọi độc lập; term được chèn vào câu nên
  ngữ pháp quanh term có thể lệch nhẹ so với câu mà model tự viết.
- Chưa có consumer nào trong repo đọc `en_*`: `common/data.py` mới validate
  `vi_*`. Output hiện là dữ liệu chuẩn bị cho thí nghiệm song ngữ.
- Chưa có lần chạy thật với GPU/model: phần sinh văn bản chưa được kiểm chứng
  bằng số liệu, chỉ phần dữ liệu và span đã được dry-run.
