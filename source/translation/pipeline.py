"""Điều phối pipeline dịch local: đọc, batch, retry, OOM, resume, ghi output."""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .llm_local import (
    HFBackend,
    build_prompt,
    format_retry_feedback,
    is_oom,
)
from .parse import (
    ModelOutputError,
    TagParseError,
    parse_model_json,
    parse_tagged,
)
from .tagging import Entity, TaggedSentence, TaggingError, attach_tags
from .umls_hints import load_hints
from .validate import SapBertMatcher, validate_translation

LOGGER = logging.getLogger(__name__)
OOM_SENTINEL = object()


class PipelineError(RuntimeError):
    """Lỗi cấu hình hoặc trạng thái không thể tiếp tục."""


@dataclass(frozen=True)
class Record:
    index: int
    raw: str
    data: dict[str, Any] | None
    entities: tuple[Entity, ...]
    vi_sentence_str: str
    error: str | None = None


@dataclass(frozen=True)
class PipelineConfig:
    model: str
    batch_size: int
    umls_path: str
    sap_thr: float
    input_path: str
    output_path: str
    review_path: str
    max_new_tokens: int = 2048
    max_seq_len: int | None = None
    dtype: str = "bfloat16"
    umls_cache: str | None = None
    sap_model: str | None = None
    max_retries: int = 2
    resume: bool = True
    limit: int | None = None


@dataclass
class RunStats:
    input_records: int = 0
    skipped: int = 0
    processed: int = 0
    translated: int = 0
    first_try: int = 0
    after_retry: int = 0
    review: int = 0
    oom_batches: int = 0
    attempts: int = 0
    elapsed_s: float = 0.0
    review_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def sentences_per_sec(self) -> float:
        return self.processed / self.elapsed_s if self.elapsed_s > 0 else 0.0

    @property
    def attempts_per_sec(self) -> float:
        return self.attempts / self.elapsed_s if self.elapsed_s > 0 else 0.0

    def as_dict(self) -> dict[str, Any]:
        total = self.translated + self.review
        return {
            "input_records": self.input_records,
            "skipped": self.skipped,
            "processed": self.processed,
            "translated": self.translated,
            "first_try": self.first_try,
            "after_retry": self.after_retry,
            "review": self.review,
            "oom_batches": self.oom_batches,
            "attempts": self.attempts,
            "elapsed_s": round(self.elapsed_s, 3),
            "sentences_per_sec": round(self.sentences_per_sec, 4),
            "attempts_per_sec": round(self.attempts_per_sec, 4),
            "rates": {
                "pass": self.translated / total if total else 0.0,
                "first_try": self.first_try / total if total else 0.0,
                "after_retry": self.after_retry / total if total else 0.0,
                "review": self.review / total if total else 0.0,
            },
            "review_reasons": dict(sorted(self.review_reasons.items())),
        }


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _parse_record(index: int, raw: str) -> Record:
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("record must be a JSON object")

        sentence = data.get("vi_sentence_str")
        if not isinstance(sentence, str) or not sentence.strip():
            raise ValueError("vi_sentence_str must be a non-empty string")

        raw_entities = data.get("entities")
        if not isinstance(raw_entities, list):
            raise ValueError("entities must be a list")

        entities: list[Entity] = []
        for entity_index, item in enumerate(raw_entities):
            if not isinstance(item, dict):
                raise ValueError(f"entity {entity_index} must be an object")
            cui = item.get("cui")
            vi_term = item.get("vi_term")
            label = item.get("label")
            start = item.get("vi_start_token_idx")
            end = item.get("vi_end_token_idx")
            icd10 = item.get("icd10", [])
            if not isinstance(cui, str) or not cui.strip():
                raise ValueError(f"entity {entity_index}: cui must be a non-empty string")
            if not isinstance(vi_term, str) or not vi_term.strip():
                raise ValueError(f"entity {entity_index}: vi_term must be a non-empty string")
            if not isinstance(label, str) or not label.strip():
                raise ValueError(f"entity {entity_index}: label must be a non-empty string")
            if not _is_int(start) or not _is_int(end):
                raise ValueError(
                    f"entity {entity_index}: token indices must be integers"
                )
            if not isinstance(icd10, list) or not all(
                isinstance(code, str) for code in icd10
            ):
                raise ValueError(f"entity {entity_index}: icd10 must be a list of strings")
            entities.append(
                Entity(
                    cui=cui.strip(),
                    icd10=tuple(icd10),
                    label=label,
                    vi_term=vi_term,
                    vi_start_token_idx=start,
                    vi_end_token_idx=end,
                )
            )

        return Record(
            index=index,
            raw=raw,
            data=data,
            entities=tuple(entities),
            vi_sentence_str=sentence,
        )
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        return Record(
            index=index,
            raw=raw,
            data=None,
            entities=(),
            vi_sentence_str="",
            error=f"{type(exc).__name__}: {exc}",
        )


def load_records(path: str | Path, limit: int | None = None) -> list[Record]:
    records: list[Record] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            raw = line.rstrip("\r\n")
            if not raw.strip():
                continue
            records.append(_parse_record(len(records), raw))
            if limit is not None and len(records) >= limit:
                break
    return records


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _resume_keys(output_path: str | Path, manifest_path: str | Path) -> set[tuple[int, str]]:
    output = Path(output_path)
    manifest = Path(manifest_path)
    if not output.exists() or output.stat().st_size == 0:
        if manifest.exists() and manifest.stat().st_size > 0:
            raise PipelineError(
                f"{manifest} is non-empty but {output} is empty/missing; "
                "move/delete the stale manifest or run with --no-resume"
            )
        return set()
    if not manifest.exists():
        raise PipelineError(
            f"{output} exists but {manifest} is missing; "
            "move/delete the old output or run with --no-resume"
        )
    if manifest.stat().st_mtime_ns < output.stat().st_mtime_ns:
        raise PipelineError(
            f"{manifest} is older than {output}; output may have a partial row. "
            "Move/delete both and rerun."
        )

    output_indices: set[int] = set()
    with output.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PipelineError(f"invalid output row in {output}") from exc
            index = row.get("record_index") if isinstance(row, dict) else None
            if not _is_int(index):
                raise PipelineError(
                    f"{output} has a row without integer record_index; "
                    "move/delete it and rerun"
                )
            if index in output_indices:
                raise PipelineError(f"duplicate record_index {index} in {output}")
            output_indices.add(index)

    keys: set[tuple[int, str]] = set()
    manifest_indices: set[int] = set()
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                index = item["index"]
                digest = item["sha1"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise PipelineError(f"invalid resume manifest line: {line!r}") from exc
            if not _is_int(index) or not isinstance(digest, str):
                raise PipelineError(f"invalid resume manifest entry: {item!r}")
            if index not in output_indices:
                raise PipelineError(
                    f"manifest index {index} is missing from {output}"
                )
            manifest_indices.add(index)
            keys.add((index, digest))
    if output_indices != manifest_indices:
        raise PipelineError(
            f"{output} and {manifest} disagree on completed indices; "
            "a crash may have left a partial row. Move/delete both or use --no-resume."
        )
    return keys


class OutputWriter:
    def __init__(
        self,
        output_path: str | Path,
        review_path: str | Path,
        *,
        truncate: bool = False,
    ) -> None:
        self.output_path = Path(output_path)
        self.review_path = Path(review_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.review_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path = Path(str(self.output_path) + ".done.jsonl")
        mode = "w" if truncate else "a"
        self._output = self.output_path.open(mode, encoding="utf-8")
        self._review = self.review_path.open(mode, encoding="utf-8")
        self._manifest = self.manifest_path.open(mode, encoding="utf-8")

    def write_translated(self, index: int, raw: str, row: dict[str, Any]) -> None:
        self._output.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._output.flush()
        manifest_row = {"index": index, "sha1": _sha1(raw)}
        self._manifest.write(json.dumps(manifest_row, ensure_ascii=False) + "\n")
        self._manifest.flush()

    def write_review(self, row: dict[str, Any]) -> None:
        self._review.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._review.flush()

    def flush(self) -> None:
        self._output.flush()
        self._review.flush()
        self._manifest.flush()

    def close(self) -> None:
        self._output.close()
        self._review.close()
        self._manifest.close()


def _reason_code(reasons: Iterable[str]) -> str:
    for reason in reasons:
        return reason.split(":", 1)[0].strip() or "unknown"
    return "unknown"


def _write_review(
    writer: OutputWriter,
    stats: RunStats,
    record: Record,
    reasons: list[str],
    *,
    reason: str,
    tagged: TaggedSentence | None = None,
    attempts: list[dict[str, Any]] | None = None,
) -> None:
    stats.review += 1
    code = _reason_code(reasons)
    stats.review_reasons[code] = stats.review_reasons.get(code, 0) + 1
    writer.write_review(
        {
            "record_index": record.index,
            "reason": reason,
            "reasons": list(reasons),
            "record": record.data,
            "raw_line": record.raw,
            "vi_tagged": tagged.vi_tagged if tagged is not None else None,
            "attempts": attempts or [],
        }
    )


def _translated_row(
    record: Record,
    clean_sentence: str,
    validation,
) -> dict[str, Any]:
    if len(validation.entities) != len(record.entities):
        raise PipelineError(
            f"record {record.index}: validated {len(validation.entities)} entities, "
            f"expected {len(record.entities)}"
        )
    entities: list[dict[str, Any]] = []
    for source, translated in zip(record.entities, validation.entities):
        entities.append(
            {
                "cui": source.cui,
                "icd10": list(source.icd10),
                "label": source.label,
                "en_term": translated.en_term,
                "en_start_token_idx": translated.en_start_token_idx,
                "en_end_token_idx": translated.en_end_token_idx,
            }
        )
    return {
        "record_index": record.index,
        "entities": entities,
        "en_sentence_str": clean_sentence,
    }


def _generate_with_oom(
    backend, prompts: list[str], state: dict[str, int]
) -> list[object]:
    """Sinh batch; OOM ở một prompt trả sentinel thay vì làm mất nửa đã xong."""

    try:
        return list(backend.generate_batch(prompts))
    except Exception as exc:
        if not is_oom(exc):
            raise
        state["batch_size"] = max(1, len(prompts) // 2)
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass
        if len(prompts) <= 1:
            return [OOM_SENTINEL]
        middle = len(prompts) // 2
        left = _generate_with_oom(backend, prompts[:middle], state)
        right = _generate_with_oom(backend, prompts[middle:], state)
        return left + right


def run_pipeline(
    config: PipelineConfig,
    *,
    backend=None,
    sap_matcher: SapBertMatcher | None = None,
) -> RunStats:
    """Chạy pipeline; không gọi API ngoài, không tự tải model."""

    started = time.perf_counter()
    stats = RunStats()
    records = load_records(config.input_path, limit=config.limit)
    stats.input_records = len(records)

    done = (
        _resume_keys(config.output_path, str(config.output_path) + ".done.jsonl")
        if config.resume
        else set()
    )

    done_indices = {index for index, _digest in done}
    pending: list[Record] = []
    for record in records:
        digest = _sha1(record.raw)
        if (record.index, digest) in done:
            stats.skipped += 1
            continue
        if record.index in done_indices:
            raise PipelineError(
                f"record_index {record.index} was translated from different "
                "input content; use --no-resume after moving/deleting old output"
            )
        pending.append(record)
    stats.processed = len(pending)

    writer = OutputWriter(
        config.output_path,
        config.review_path,
        truncate=not config.resume,
    )
    try:
        valid: list[Record] = []
        for record in pending:
            if record.error is not None:
                _write_review(
                    writer,
                    stats,
                    record,
                    [f"bad_record: {record.error}"],
                    reason="bad_record",
                )
            else:
                valid.append(record)

        tagged: dict[int, TaggedSentence] = {}
        ready: list[Record] = []
        for record in valid:
            try:
                tagged[record.index] = attach_tags(
                    record.vi_sentence_str, record.entities
                )
            except TaggingError as exc:
                _write_review(
                    writer,
                    stats,
                    record,
                    [f"{exc.code}: {exc.detail}"],
                    reason="tagging_failed",
                )
            else:
                ready.append(record)

        cuis = {entity.cui for record in ready for entity in record.entities}
        hints = load_hints(
            config.umls_path,
            cuis,
            cache_path=config.umls_cache,
        )

        if ready:
            if backend is None:
                backend = HFBackend(
                    config.model,
                    max_new_tokens=config.max_new_tokens,
                    max_seq_len=config.max_seq_len,
                    dtype=config.dtype,
                )
            if sap_matcher is None:
                candidate = SapBertMatcher(config.sap_model, config.sap_thr)
                if candidate.available:
                    sap_matcher = candidate
                else:
                    LOGGER.warning(
                        "SapBERT unavailable (no local model or missing "
                        "sentence-transformers); CUI validation is exact-only"
                    )
                    sap_matcher = None

        state = {"batch_size": max(1, config.batch_size)}
        feedback: dict[int, str] = {}
        attempt_log: dict[int, list[dict[str, Any]]] = {}

        for attempt in range(config.max_retries + 1):
            if not ready:
                break
            ready.sort(key=lambda item: len(item.vi_sentence_str))
            next_ready: list[Record] = []

            position = 0
            while position < len(ready):
                size = max(1, state["batch_size"])
                chunk = ready[position:position + size]
                position += len(chunk)
                prompts = [
                    build_prompt(
                        record.entities,
                        tagged[record.index].vi_tagged,
                        hints,
                        feedback.get(record.index, ""),
                    )
                    for record in chunk
                ]
                try:
                    raws = _generate_with_oom(backend, prompts, state)
                except Exception as exc:
                    for record in chunk:
                        _write_review(
                            writer,
                            stats,
                            record,
                            [f"backend_error: {exc}"],
                            reason="backend_error",
                            tagged=tagged[record.index],
                            attempts=attempt_log.get(record.index, []),
                        )
                    continue

                if len(raws) != len(chunk):
                    for record in chunk:
                        _write_review(
                            writer,
                            stats,
                            record,
                            [
                                "backend_error: model returned "
                                f"{len(raws)} results for {len(chunk)} prompts"
                            ],
                            reason="backend_error",
                            tagged=tagged[record.index],
                            attempts=attempt_log.get(record.index, []),
                        )
                    continue

                for record, raw in zip(chunk, raws):
                    stats.attempts += 1
                    attempt_entry = {"attempt": attempt + 1, "reasons": []}
                    attempt_log.setdefault(record.index, []).append(attempt_entry)
                    if raw is OOM_SENTINEL:
                        stats.oom_batches += 1
                        reasons = [
                            "oom: single prompt does not fit in memory"
                        ]
                        attempt_entry["reasons"] = reasons
                        _write_review(
                            writer,
                            stats,
                            record,
                            reasons,
                            reason="oom",
                            tagged=tagged[record.index],
                            attempts=attempt_log.get(record.index, []),
                        )
                        continue
                    retryable = True
                    try:
                        payload = parse_model_json(str(raw))
                        parsed = parse_tagged(str(payload["en_tagged"]))
                        validation = validate_translation(
                            record.entities,
                            parsed,
                            hints,
                            sap_matcher,
                        )
                    except (ModelOutputError, TagParseError) as exc:
                        reasons = [f"{exc.code}: {exc.detail}"]
                    except Exception as exc:
                        reasons = [f"internal_error: {exc}"]
                    else:
                        if validation.ok:
                            row = _translated_row(
                                record, parsed.clean, validation
                            )
                            writer.write_translated(record.index, record.raw, row)
                            stats.translated += 1
                            if attempt == 0:
                                stats.first_try += 1
                            else:
                                stats.after_retry += 1
                            continue
                        reasons = list(validation.reasons)
                        retryable = validation.retryable

                    attempt_entry["reasons"] = reasons
                    if attempt < config.max_retries and retryable:
                        feedback[record.index] = format_retry_feedback(reasons)
                        next_ready.append(record)
                    else:
                        _write_review(
                            writer,
                            stats,
                            record,
                            reasons,
                            reason="validation_failed",
                            tagged=tagged[record.index],
                            attempts=attempt_log.get(record.index, []),
                        )

                writer.flush()

            ready = next_ready

        # Phòng trường hợp bất thường: record còn lại chưa có kết quả.
        for record in ready:
            _write_review(
                writer,
                stats,
                record,
                ["validation_failed: retry budget exhausted"],
                reason="validation_failed",
                tagged=tagged[record.index],
                attempts=attempt_log.get(record.index, []),
            )

        stats.elapsed_s = time.perf_counter() - started
        return stats
    finally:
        writer.close()
