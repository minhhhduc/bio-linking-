"""Đọc MRCONSO và lấy tối đa 8 tên tiếng Anh cho mỗi CUI.

Cột MRCONSO chuẩn:
``CUI|LAT|TS|LUI|STT|SUI|ISPREF|AUI|SAUI|SCUI|SDUI|SAB|TTY|CODE|STR|SRL|SUPPRESS``
"""

from __future__ import annotations

import gzip
import os
import pickle
import unicodedata
from pathlib import Path
from typing import Iterable

CACHE_VERSION = 2
COL_CUI = 0
COL_LAT = 1
COL_TS = 2
COL_ISPREF = 6
COL_TTY = 12
COL_STR = 14
COL_SUPPRESS = 16
MIN_COLUMNS = COL_SUPPRESS + 1


def _normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def _cache_key(
    umls_path: Path,
    cuis: frozenset[str],
    max_names: int,
) -> tuple[object, ...]:
    stat = umls_path.stat()
    return (
        CACHE_VERSION,
        str(umls_path.resolve()),
        stat.st_mtime_ns,
        stat.st_size,
        tuple(sorted(cuis)),
        max_names,
    )


def _load_cache(cache_path: Path, key: tuple[object, ...]) -> dict[str, tuple[str, ...]] | None:
    try:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
    except (OSError, EOFError, pickle.PickleError, AttributeError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("key") != key:
        return None
    hints = payload.get("hints")
    if not isinstance(hints, dict):
        return None
    if not all(
        isinstance(value, tuple)
        and value
        and all(isinstance(name, str) for name in value)
        for value in hints.values()
    ):
        return None
    return hints


def _save_cache(
    cache_path: Path,
    key: tuple[object, ...],
    hints: dict[str, tuple[str, ...]],
) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
        with temporary.open("wb") as handle:
            pickle.dump({"key": key, "hints": hints}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary, cache_path)
    except (OSError, pickle.PickleError):
        # Cache chỉ là tối ưu; không được làm hỏng lần chạy.
        return


def _is_gzip(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(2) == b"\x1f\x8b"
    except OSError:
        return False


def _open_text(path: Path):
    if path.suffix.lower() == ".gz" or _is_gzip(path):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("rt", encoding="utf-8", errors="replace")


def load_hints(
    umls_path: str | Path,
    cuis: Iterable[str],
    *,
    max_names: int = 8,
    cache_path: str | Path | None = None,
) -> dict[str, tuple[str, ...]]:
    """Trả ``{cui: (name, ...)}`` đã xếp hạng, tối đa ``max_names`` tên/CUI."""

    path = Path(umls_path)
    if not path.exists():
        raise FileNotFoundError(f"UMLS MRCONSO not found: {path}")

    wanted = frozenset(cui.strip() for cui in cuis if isinstance(cui, str) and cui.strip())
    if not wanted:
        return {}
    if max_names <= 0:
        raise ValueError("max_names must be positive")

    key = _cache_key(path, wanted, max_names)
    cache_file = Path(cache_path) if cache_path is not None else None
    if cache_file is not None:
        cached = _load_cache(cache_file, key)
        if cached is not None:
            return cached

    # (ưu tiên ISPREF=Y, TS=P, TTY PT/MH, rồi tên ngắn hơn/alphabet)
    candidates: dict[str, list[tuple[int, int, int, int, str, str]]] = {}
    with _open_text(path) as handle:
        for line in handle:
            if not line.strip():
                continue
            fields = line.rstrip("\r\n").split("|")
            if len(fields) < MIN_COLUMNS:
                continue
            cui = fields[COL_CUI]
            if cui not in wanted:
                continue
            if fields[COL_LAT] != "ENG":
                continue
            if fields[COL_SUPPRESS].strip() == "Y":
                continue
            name = fields[COL_STR].strip()
            if not name or "\ufffd" in name:
                continue
            ispref_rank = 0 if fields[COL_ISPREF] == "Y" else 1
            ts_rank = 0 if fields[COL_TS] == "P" else 1
            tty_rank = 0 if fields[COL_TTY] in {"PT", "MH"} else 1
            candidates.setdefault(cui, []).append(
                (ispref_rank, ts_rank, tty_rank, len(name), name.casefold(), name)
            )

    hints: dict[str, tuple[str, ...]] = {}
    for cui, rows in candidates.items():
        rows.sort()
        selected: list[str] = []
        seen: set[str] = set()
        for _ispref, _ts, _tty, _length, _folded, name in rows:
            normalized = _normalize(name)
            if normalized in seen:
                continue
            seen.add(normalized)
            selected.append(name)
            if len(selected) >= max_names:
                break
        if selected:
            hints[cui] = tuple(selected)

    if cache_file is not None:
        _save_cache(cache_file, key, hints)
    return hints
