"""C4: exact cosine vector index over KB surface forms.

Vectors are stored L2-normalised as float32, so a dot product is cosine
similarity. Persistence is ``vectors.npy`` + ``meta.json`` (no pickle).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .kb import surface_entries
from .schemas import Concept, RetrievalHit


class VectorIndex:
    """One entry per KB surface form; exact cosine scan at query time."""

    def __init__(
        self,
        vectors: np.ndarray,
        cuis: Sequence[str],
        surfaces: Sequence[str],
        source_kbs: Sequence[str | None] | None = None,
        *,
        encoder_info: Mapping[str, str] | None = None,
    ) -> None:
        array = np.array(vectors, dtype=np.float32, copy=True)
        if array.ndim != 2:
            raise ValueError(f"vectors must be 2-D, got shape {array.shape!r}")
        if array.shape[1] < 1:
            raise ValueError("vectors must have at least one embedding dimension")

        if isinstance(cuis, (str, bytes)):
            raise ValueError("cuis must be a sequence of strings, not a string")
        if isinstance(surfaces, (str, bytes)):
            raise ValueError("surfaces must be a sequence of strings, not a string")
        if isinstance(source_kbs, (str, bytes)):
            raise ValueError("source_kbs must be a sequence, not a string")

        cuis = tuple(cuis)
        surfaces = tuple(surfaces)
        count = array.shape[0]
        if not (len(cuis) == len(surfaces) == count):
            raise ValueError(
                "vectors, cuis and surfaces must have equal length "
                f"(got {count}, {len(cuis)}, {len(surfaces)})"
            )

        if source_kbs is None:
            source_kbs = (None,) * count
        else:
            source_kbs = tuple(source_kbs)
            if len(source_kbs) != count:
                raise ValueError(
                    "source_kbs must have the same length as vectors "
                    f"(got {len(source_kbs)} and {count})"
                )

        if encoder_info is not None:
            if not isinstance(encoder_info, Mapping):
                raise ValueError("encoder_info must be a mapping or None")
            normalized_info: dict[str, str] = {}
            for key, value in encoder_info.items():
                if not isinstance(key, str) or not key.strip():
                    raise ValueError("encoder_info keys must be non-empty strings")
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(
                        "encoder_info values must be non-empty strings"
                    )
                normalized_info[key] = value
            encoder_info = normalized_info

        for cui in cuis:
            if not isinstance(cui, str) or not cui.strip():
                raise ValueError("every cui must be a non-empty string")
        for surface in surfaces:
            if not isinstance(surface, str) or not surface.strip():
                raise ValueError("every surface must be a non-empty string")
        for source_kb in source_kbs:
            if source_kb is not None and (
                not isinstance(source_kb, str) or not source_kb.strip()
            ):
                raise ValueError("source_kb must be None or a non-empty string")
        if len(set(zip(cuis, surfaces, source_kbs))) != count:
            raise ValueError(
                "entries must be unique (cui, surface, source_kb) triples; "
                "duplicate surface entries waste Top-k slots"
            )

        if not np.all(np.isfinite(array)):
            raise ValueError("vectors contain non-finite values")
        if count:
            norms = np.linalg.norm(array, axis=1, keepdims=True)
            if not np.all(np.isfinite(norms)) or np.any(norms <= 0.0):
                raise ValueError("vectors contain a zero-norm row")
            array = array / norms

        self.vectors = array.astype(np.float32, copy=False)
        self.cuis = cuis
        self.surfaces = surfaces
        self.source_kbs = source_kbs
        self.encoder_info = encoder_info

    @classmethod
    def from_concepts(
        cls,
        concepts: Iterable[Concept],
        encoder,
        *,
        batch_size: int = 64,
        encoder_info: Mapping[str, str] | None = None,
    ) -> "VectorIndex":
        """Index every surface form produced by ``kb.surface_entries``."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if isinstance(concepts, (str, bytes)):
            raise ValueError("concepts must be an iterable of Concept records")

        entries = list(surface_entries(list(concepts)))
        if not entries:
            return cls(
                np.zeros((0, encoder.dim), dtype=np.float32),
                (),
                (),
                encoder_info=encoder_info,
            )

        cuis = [entry[0] for entry in entries]
        surfaces = [entry[1] for entry in entries]
        source_kbs = [entry[2] for entry in entries]

        batches: list[np.ndarray] = []
        expected_dim: int | None = None
        for start in range(0, len(surfaces), batch_size):
            chunk = surfaces[start : start + batch_size]
            encoded = np.asarray(encoder.encode(chunk), dtype=np.float32)
            if encoded.ndim != 2 or encoded.shape[0] != len(chunk):
                raise ValueError(
                    "encoder must return (n, dim) for each batch, got "
                    f"{encoded.shape!r} for {len(chunk)} texts"
                )
            if expected_dim is None:
                expected_dim = int(encoded.shape[1])
            elif encoded.shape[1] != expected_dim:
                raise ValueError("encoder returned inconsistent dimensions")
            batches.append(encoded)

        return cls(
            np.concatenate(batches, axis=0),
            cuis,
            surfaces,
            source_kbs,
            encoder_info=encoder_info,
        )

    def __len__(self) -> int:
        return int(self.vectors.shape[0])

    def search(
        self,
        query,
        top_k: int,
        *,
        source_kb: str | None = None,
    ) -> list[RetrievalHit]:
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, (int, np.integer))
            or top_k < 1
        ):
            raise ValueError("top_k must be a positive integer")
        if source_kb is not None and (
            not isinstance(source_kb, str) or not source_kb.strip()
        ):
            raise ValueError("source_kb must be None or a non-empty string")

        vector = np.asarray(query, dtype=np.float32)
        if vector.ndim != 1:
            raise ValueError(f"query must be 1-D, got shape {vector.shape!r}")
        if not np.all(np.isfinite(vector)):
            raise ValueError("query contains non-finite values")
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError("query must have a non-zero norm")

        if vector.shape[0] != self.vectors.shape[1]:
            raise ValueError(
                f"query dimension {vector.shape[0]} does not match index "
                f"dimension {self.vectors.shape[1]}"
            )
        if len(self) == 0:
            return []

        if source_kb is None:
            indices = range(len(self))
        else:
            indices = [
                i for i, kb_name in enumerate(self.source_kbs) if kb_name == source_kb
            ]
        if not indices:
            return []

        unit_query = vector / norm
        # Exact cosine scan: docs/architecture.md C4 explicitly allows "exact"
        # as an index implementation; FAISS/HNSW can replace this class.
        scores = self.vectors @ unit_query
        ordered = sorted(
            indices,
            key=lambda i: (
                -float(scores[i]),
                self.cuis[i],
                self.surfaces[i],
            ),
        )

        hits: list[RetrievalHit] = []
        for rank, i in enumerate(ordered[:top_k], start=1):
            hits.append(
                RetrievalHit(
                    cui=self.cuis[i],
                    surface=self.surfaces[i],
                    score=float(scores[i]),
                    rank=rank,
                )
            )
        return hits

    def save(self, directory) -> None:
        base = Path(directory)
        base.mkdir(parents=True, exist_ok=True)

        np.save(base / "vectors.npy", self.vectors, allow_pickle=False)
        meta = {
            "count": len(self),
            "dim": int(self.vectors.shape[1]),
            "cuis": list(self.cuis),
            "surfaces": list(self.surfaces),
            "source_kbs": list(self.source_kbs),
            "encoder_info": self.encoder_info,
        }
        with (base / "meta.json").open("w", encoding="utf-8") as handle:
            json.dump(meta, handle, ensure_ascii=False, indent=2, sort_keys=True)

    @classmethod
    def load(cls, directory) -> "VectorIndex":
        base = Path(directory)
        vectors = np.load(base / "vectors.npy", allow_pickle=False)
        with (base / "meta.json").open("r", encoding="utf-8-sig") as handle:
            meta = json.load(handle)

        if not isinstance(meta, dict):
            raise ValueError("meta.json must contain a JSON object")
        required = ("count", "dim", "cuis", "surfaces", "source_kbs")
        missing = [key for key in required if key not in meta]
        if missing:
            raise ValueError(f"meta.json is missing keys: {', '.join(missing)}")

        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError(f"saved vectors must be 2-D, got {vectors.shape!r}")

        count = meta["count"]
        dim = meta["dim"]
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or isinstance(dim, bool)
            or not isinstance(dim, int)
        ):
            raise ValueError("meta.json count/dim must be integers")
        if count != vectors.shape[0] or dim != vectors.shape[1]:
            raise ValueError(
                "meta.json disagrees with vectors.npy shape "
                f"({count=}, {dim=}, {vectors.shape=})"
            )

        cuis = meta["cuis"]
        surfaces = meta["surfaces"]
        source_kbs = meta["source_kbs"]
        for name, values in (
            ("cuis", cuis),
            ("surfaces", surfaces),
            ("source_kbs", source_kbs),
        ):
            if not isinstance(values, list) or len(values) != count:
                raise ValueError(
                    f"meta.json {name!r} must be a list of length {count}"
                )

        return cls(
            vectors,
            cuis,
            surfaces,
            source_kbs,
            encoder_info=meta.get("encoder_info"),
        )
