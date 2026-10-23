"""KB loading (C1) and homonym rewriting (C2) — architecture §3, §5.1.

The index-building contract is one ``(cui, surface, source_kb)`` entry per
distinct surface form per concept.  An alias that occurs in the alias lists of
several distinct CUIs is rewritten for every alias occurrence as
``f"{alias} ({concept.name})"``; preferred names stay unchanged
(architecture §5.1, reference_abstract §3).
"""

from __future__ import annotations

import csv
import json
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .schemas import Concept


def _normalize(text: str) -> str:
    """NFC-normalise, collapse whitespace runs, and strip the ends."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def is_homonym_surface(a: str, b: str) -> bool:
    """Return whether two surface forms are equal after NFC/whitespace normalisation."""
    return _normalize(a) == _normalize(b)


def load_concepts_jsonl(
    path: str | Path, *, progress_callback=None, total: int | None = None
) -> list[Concept]:
    """Load a concept KB from JSON Lines.

    Each non-blank line must be a JSON object with non-empty string ``cui`` and
    ``name``, an optional list of non-empty string ``aliases``, and an optional
    non-empty string ``source_kb``.  Blank lines are ignored.  Invalid input,
    including a repeated ``(cui, source_kb)`` pair (a concept record already
    carries all of its aliases), raises ``ValueError`` naming the offending
    line.  The same CUI may appear once per source KB so that the index can be
    filtered by ``source_kb``.  ``progress_callback(done, total)`` is called
    after each concept is parsed; pass ``total`` when the caller has already
    counted the records, otherwise it is ``None``.
    """
    if progress_callback is not None and not callable(progress_callback):
        raise ValueError("progress_callback must be callable or None")
    if total is not None and (
        isinstance(total, bool) or not isinstance(total, int) or total < 0
    ):
        raise ValueError("total must be a non-negative integer or None")

    path = Path(path)
    concepts: list[Concept] = []
    seen_concepts: dict[tuple[str, str | None], int] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{lineno}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(raw, dict):
                raise ValueError(
                    f"{path}:{lineno}: concept record must be a JSON object"
                )

            cui = raw.get("cui")
            if not isinstance(cui, str) or not cui.strip():
                raise ValueError(
                    f"{path}:{lineno}: 'cui' must be a non-empty string"
                )
            cui = unicodedata.normalize("NFC", cui.strip())
            name = raw.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(
                    f"{path}:{lineno}: 'name' must be a non-empty string"
                )
            name = name.strip()

            aliases_raw = raw.get("aliases", [])
            if aliases_raw is None:
                aliases_raw = []
            if not isinstance(aliases_raw, list):
                raise ValueError(
                    f"{path}:{lineno}: 'aliases' must be a list of strings"
                )
            aliases: list[str] = []
            for index, alias in enumerate(aliases_raw):
                if not isinstance(alias, str) or not alias.strip():
                    raise ValueError(
                        f"{path}:{lineno}: aliases[{index}] must be a "
                        "non-empty string"
                    )
                aliases.append(alias)

            source_kb = raw.get("source_kb")
            if source_kb is not None and (
                not isinstance(source_kb, str) or not source_kb.strip()
            ):
                raise ValueError(
                    f"{path}:{lineno}: 'source_kb' must be a non-empty "
                    "string when present"
                )
            if source_kb is not None:
                source_kb = source_kb.strip()

            concept_key = (cui, source_kb)
            if concept_key in seen_concepts:
                raise ValueError(
                    f"{path}:{lineno}: duplicate cui {cui!r} for "
                    f"source_kb {source_kb!r} "
                    f"(first seen on line {seen_concepts[concept_key]})"
                )
            seen_concepts[concept_key] = lineno

            concepts.append(
                Concept(
                    cui=cui,
                    name=name,
                    aliases=tuple(aliases),
                    source_kb=source_kb,
                )
            )
            if progress_callback is not None:
                progress_callback(len(concepts), total)
    return concepts


def load_concepts_csv(
    path: str | Path, *, progress_callback=None, total: int | None = None
) -> list[Concept]:
    """Load a two-column ``CUI,concept`` CSV into concepts.

    The header must be ``CUI,concept`` (case-insensitive).  Rows are grouped by
    CUI: the first ``concept`` is the preferred name and later distinct
    concepts become aliases in first-appearance order, so a repeated CUI does
    not abort a large KB load.  A concept spelt ``"NaN"`` is real KB content
    (the "none/no" concept, e.g. C0549184), not a missing value, and is kept
    like any other concept.  Only a genuinely empty field is rejected.
    ``progress_callback(done, total)`` is called after each parsed row; pass
    ``total`` when the caller already counted the rows.
    """
    if progress_callback is not None and not callable(progress_callback):
        raise ValueError("progress_callback must be callable or None")
    if total is not None and (
        isinstance(total, bool) or not isinstance(total, int) or total < 0
    ):
        raise ValueError("total must be a non-negative integer or None")

    path = Path(path)
    order: list[str] = []
    names: dict[str, str] = {}
    aliases: dict[str, list[str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(
                f"{path}: empty CSV; expected a 'CUI,concept' header"
            ) from None
        if [column.strip().lower() for column in header] != ["cui", "concept"]:
            raise ValueError(
                f"{path}: expected header 'CUI,concept', got {header!r}"
            )

        done = 0
        for lineno, row in enumerate(reader, 2):
            if not row or all(not field.strip() for field in row):
                continue
            if len(row) != 2:
                raise ValueError(
                    f"{path}:{lineno}: expected 2 columns, got {len(row)}"
                )
            cui, concept = row
            if not cui.strip():
                raise ValueError(f"{path}:{lineno}: 'CUI' must be non-empty")
            if not concept.strip():
                raise ValueError(f"{path}:{lineno}: 'concept' must be non-empty")
            cui = unicodedata.normalize("NFC", cui.strip())
            concept = unicodedata.normalize("NFC", concept.strip())

            if cui in names:
                if concept != names[cui] and concept not in aliases[cui]:
                    aliases[cui].append(concept)
            else:
                order.append(cui)
                names[cui] = concept
                aliases[cui] = []
            done += 1
            if progress_callback is not None:
                progress_callback(done, total)

    return [
        Concept(cui=cui, name=names[cui], aliases=tuple(aliases[cui]), source_kb=None)
        for cui in order
    ]


def concepts_from_dataset(
    records: Iterable[Mapping[str, Any]],
) -> list[Concept]:
    """Derive a surrogate KB from gold BEL annotations.

    ``name`` is the most frequent ``vi_term`` observed for a CUI (ties go to
    the term that appeared first); ``aliases`` are the remaining distinct
    terms in first-appearance order; ``source_kb`` is ``None``.

    LEAKAGE WARNING: this KB is derived **from the gold annotations
    themselves**, not from a user-supplied KB.  Every gold mention is
    therefore indexed under its own gold
    CUI.  Using this surrogate to evaluate the same file gives mention/alias
    leakage and inflates retrieval and accuracy; it is suitable only for
    plumbing tests, smoke runs, and demos on held-out mentions.

    Records are the parsed objects of ``test_fixed_verified_gold_revised.jsonl``
    (``{"entities": [...], "vi_sentence_str": ...}``).  Malformed records and
    entities are skipped rather than raising; this function never invents
    concepts without a usable ``cui`` and ``vi_term``.
    """
    if isinstance(records, (str, bytes, Mapping)):
        raise ValueError("records must be an iterable of record mappings")

    order: list[str] = []
    counts: dict[str, Counter[str]] = {}
    first_seen: dict[str, dict[str, int]] = {}
    occurrence = 0

    for record in records:
        if not isinstance(record, Mapping):
            continue
        entities = record.get("entities")
        if not isinstance(entities, list):
            continue
        for entity in entities:
            if not isinstance(entity, Mapping):
                continue
            cui = entity.get("cui")
            term = entity.get("vi_term")
            if not isinstance(cui, str) or not cui.strip():
                continue
            if not isinstance(term, str) or not term.strip():
                continue
            cui = unicodedata.normalize("NFC", cui.strip())
            term = _normalize(term)

            if cui not in counts:
                order.append(cui)
                counts[cui] = Counter()
                first_seen[cui] = {}
            if term not in first_seen[cui]:
                first_seen[cui][term] = occurrence
            counts[cui][term] += 1
            occurrence += 1

    concepts: list[Concept] = []
    for cui in order:
        cui_counts = counts[cui]
        # Highest count wins; min() resolves ties by first appearance.
        name = min(cui_counts, key=lambda term: (-cui_counts[term],
                                                 first_seen[cui][term]))
        aliases = tuple(
            term for term in first_seen[cui] if term != name
        )
        concepts.append(Concept(cui=cui, name=name, aliases=aliases,
                                source_kb=None))
    return concepts


def surface_entries(
    concepts: Iterable[Concept],
    *,
    progress_callback=None,
) -> list[tuple[str, str, str | None]]:
    """Expand concepts into ``(cui, surface, source_kb)`` index entries.

    The preferred name is always emitted first for each concept.  Aliases are
    deduplicated per concept after NFC/whitespace normalisation, and an alias
    equal to the preferred name is not emitted twice.  An alias that occurs in
    the alias lists of more than one distinct CUI is rewritten for every alias
    occurrence as ``"<surface> (<preferred name>)"``; the same CUI repeated
    across source KBs counts once.  Preferred names are indexed unchanged and
    are not counted for this rule, matching the alias-intersection definition
    in reference_abstract §3 and the alias-only rule in architecture §5.1.
    Rewritten forms are deduplicated again so a rewrite cannot create a
    duplicate entry for the same concept.  Ordering is deterministic: concept
    input order, preferred name before aliases, aliases in their input order.
    """
    if progress_callback is not None and not callable(progress_callback):
        raise ValueError("progress_callback must be callable or None")
    if isinstance(concepts, (str, bytes)):
        raise ValueError("concepts must be an iterable of Concept records")

    concept_list = list(concepts)
    expanded: list[tuple[Concept, list[tuple[str, bool]]]] = []
    alias_concepts: dict[str, set[str]] = {}

    for done, concept in enumerate(concept_list, start=1):
        listed = (concept.name, *concept.aliases)
        if isinstance(concept.aliases, str):
            raise ValueError("Concept.aliases must be a sequence, not a string")
        seen: set[str] = set()
        surfaces: list[tuple[str, bool]] = []
        for index, raw in enumerate(listed):
            if not isinstance(raw, str):
                continue
            surface = _normalize(raw)
            if not surface or surface in seen:
                continue
            seen.add(surface)
            surfaces.append((surface, index == 0))
            if index > 0:
                alias_concepts.setdefault(surface, set()).add(concept.cui)
        expanded.append((concept, surfaces))
        if progress_callback is not None:
            progress_callback(done, len(concept_list))

    entries: list[tuple[str, str, str | None]] = []
    for concept, surfaces in expanded:
        preferred = _normalize(concept.name)
        emitted: set[str] = set()
        for surface, is_preferred in surfaces:
            if not is_preferred and len(alias_concepts[surface]) > 1:
                surface = f"{surface} ({preferred})"
            if surface in emitted:
                continue
            emitted.add(surface)
            entries.append((concept.cui, surface, concept.source_kb))
    return entries
