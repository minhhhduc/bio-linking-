"""C4: exact cosine vector index over KB surface forms.

Vectors are stored L2-normalised as float32, so a dot product is cosine
similarity. Persistence is either ``vectors.npy`` + ``meta.json``
(:meth:`VectorIndex.save`/:meth:`VectorIndex.load`) or a single ``.npz`` file
(:meth:`VectorIndex.save_npz`/:meth:`VectorIndex.load_npz`); neither uses pickle.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .kb import surface_entries
from .schemas import Concept, RetrievalHit

# Written into single-file artifacts so a future format change can be detected.
SINGLE_FILE_FORMAT = "grf-vector-index-npz-v1"


class VectorIndex:
    """One entry per KB surface form; exact cosine scan at query time."""

    # Bounds the transient (N, queries) score matrix on the GPU.
    _GPU_QUERY_CHUNK = 64

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
        self._device: str | None = None
        self._gpu_vectors = None

    @classmethod
    def from_concepts(
        cls,
        concepts: Iterable[Concept],
        encoder,
        *,
        batch_size: int = 64,
        encoder_info: Mapping[str, str] | None = None,
        progress_callback=None,
        entries: Iterable[tuple[str, str, str | None]] | None = None,
    ) -> "VectorIndex":
        """Index every surface form produced by ``kb.surface_entries``.

        ``progress_callback(done, total)``, when supplied, is called after each
        encoded batch (used by the Colab notebooks to drive ``tqdm``).  Pass
        ``entries`` to reuse an expansion the caller already computed instead
        of recomputing it from ``concepts``.
        """
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if progress_callback is not None and not callable(progress_callback):
            raise ValueError("progress_callback must be callable or None")
        if isinstance(concepts, (str, bytes)):
            raise ValueError("concepts must be an iterable of Concept records")

        if entries is None:
            entries = surface_entries(list(concepts))
        entries = list(entries)
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
            if progress_callback is not None:
                progress_callback(
                    min(start + len(chunk), len(surfaces)), len(surfaces)
                )

        return cls(
            np.concatenate(batches, axis=0),
            cuis,
            surfaces,
            source_kbs,
            encoder_info=encoder_info,
        )

    def __len__(self) -> int:
        return int(self.vectors.shape[0])

    # ------------------------------------------------------------------
    # Device placement and search
    # ------------------------------------------------------------------
    @property
    def search_device(self) -> str:
        """``"cuda"`` when the vectors are cached on the GPU, else ``"cpu"``."""
        return self._device or "cpu"

    def use_gpu(self, device: str = "cuda") -> None:
        """Keep a device copy of the vectors for exact cosine scans.

        PyTorch is imported lazily (it ships with sentence-transformers), so
        the CPU path never needs it. ``device="cpu"`` releases the GPU copy.
        """
        if device == "cpu":
            self.release_gpu()
            return
        if device != "cuda":
            raise ValueError("device must be 'cuda' or 'cpu'")
        if self._device == device and self._gpu_vectors is not None:
            return
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "GPU search needs PyTorch; install sentence-transformers or torch."
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError(
                "device='cuda' but CUDA is not available; use the CPU path "
                "or enable a GPU runtime."
            )
        self._gpu_vectors = torch.from_numpy(
            np.ascontiguousarray(self.vectors)
        ).to(device)
        self._device = device

    def release_gpu(self) -> None:
        """Drop the device copy and return its cached memory to the driver."""
        self._gpu_vectors = None
        self._device = None
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _validate_top_k(top_k: int) -> None:
        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, (int, np.integer))
            or top_k < 1
        ):
            raise ValueError("top_k must be a positive integer")

    @staticmethod
    def _validate_source_kb(source_kb: str | None) -> None:
        if source_kb is not None and (
            not isinstance(source_kb, str) or not source_kb.strip()
        ):
            raise ValueError("source_kb must be None or a non-empty string")

    def _search_cpu(
        self,
        vector: np.ndarray,
        top_k: int,
        source_kb: str | None,
    ) -> list[RetrievalHit]:
        if source_kb is None:
            indices = range(len(self))
        else:
            indices = [
                i for i, kb_name in enumerate(self.source_kbs)
                if kb_name == source_kb
            ]
        if not indices:
            return []

        unit_query = vector / float(np.linalg.norm(vector))
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
        return [
            RetrievalHit(
                cui=self.cuis[i],
                surface=self.surfaces[i],
                score=float(scores[i]),
                rank=rank,
            )
            for rank, i in enumerate(ordered[:top_k], start=1)
        ]

    def search_batch(
        self,
        queries,
        top_k: int,
        *,
        source_kb: str | None = None,
    ) -> list[list[RetrievalHit]]:
        """Top-k exact cosine hits for every row of ``queries``.

        The ordering rule is identical to calling :meth:`search` per row.
        With :meth:`use_gpu` active, queries are scored in chunks on the
        device and only rows at or above each k-th score are copied back to
        CPU; float32 scores can differ in the last ULPs from the CPU path.
        """
        self._validate_top_k(top_k)
        self._validate_source_kb(source_kb)
        matrix = np.asarray(queries, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"queries must be 2-D, got shape {matrix.shape!r}")
        if matrix.shape[0] == 0:
            return []
        if matrix.shape[1] != self.vectors.shape[1]:
            raise ValueError(
                f"query dimension {matrix.shape[1]} does not match index "
                f"dimension {self.vectors.shape[1]}"
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError("queries contain non-finite values")
        norms = np.linalg.norm(matrix, axis=1)
        if not np.all(np.isfinite(norms)) or np.any(norms <= 0.0):
            raise ValueError("every query must have a non-zero norm")
        if self._device is not None:
            return self._search_gpu(matrix, top_k, source_kb)
        return [self._search_cpu(row, top_k, source_kb) for row in matrix]

    def _search_gpu(
        self,
        matrix: np.ndarray,
        top_k: int,
        source_kb: str | None,
    ) -> list[list[RetrievalHit]]:
        import torch

        empty: list[list[RetrievalHit]] = [[] for _ in range(matrix.shape[0])]
        if len(self) == 0:
            return empty
        if source_kb is None:
            positions: list[int] | None = None
            candidate_count = len(self)
        else:
            positions = [
                i for i, kb_name in enumerate(self.source_kbs)
                if kb_name == source_kb
            ]
            if not positions:
                return empty
            candidate_count = len(positions)

        device = self._device
        columns = min(top_k, candidate_count)
        positions_cpu: np.ndarray | None = (
            None if positions is None else np.asarray(positions, dtype=np.int64)
        )
        rows = (
            None
            if positions is None
            else torch.tensor(positions, dtype=torch.long, device=device)
        )

        results: list[list[RetrievalHit]] = []
        for start in range(0, matrix.shape[0], self._GPU_QUERY_CHUNK):
            chunk = matrix[start : start + self._GPU_QUERY_CHUNK]
            with torch.no_grad():
                queries = torch.from_numpy(
                    np.ascontiguousarray(chunk)
                ).to(device)
                queries = queries / queries.norm(
                    dim=1, keepdim=True
                ).clamp_min(torch.finfo(queries.dtype).tiny)
                scores = self._gpu_vectors @ queries.T
                if rows is not None:
                    scores = scores.index_select(0, rows)

                # Take the threshold and keep every row at or above it, then
                # apply the CPU order (score desc, cui, surface). Scores can
                # differ in the last float32 ULPs from BLAS, but the ordering
                # rule is the same as the CPU path.
                thresholds = torch.topk(scores, k=columns, dim=0).values[-1]

                for column in range(scores.shape[1]):
                    selected = torch.nonzero(
                        scores[:, column] >= thresholds[column], as_tuple=False
                    ).squeeze(1)
                    indices = selected.cpu().numpy()
                    values = scores[selected, column].cpu().numpy()
                    if positions_cpu is not None:
                        indices = positions_cpu[indices]
                    order = sorted(
                        range(len(indices)),
                        key=lambda j: (
                            -float(values[j]),
                            self.cuis[int(indices[j])],
                            self.surfaces[int(indices[j])],
                        ),
                    )
                    results.append([
                        RetrievalHit(
                            cui=self.cuis[int(indices[j])],
                            surface=self.surfaces[int(indices[j])],
                            score=float(values[j]),
                            rank=rank,
                        )
                        for rank, j in enumerate(order[:top_k], start=1)
                    ])
        return results

    def search(
        self,
        query,
        top_k: int,
        *,
        source_kb: str | None = None,
    ) -> list[RetrievalHit]:
        vector = np.asarray(query, dtype=np.float32)
        if vector.ndim != 1:
            raise ValueError(f"query must be 1-D, got shape {vector.shape!r}")
        return self.search_batch(
            vector[None, :], top_k, source_kb=source_kb
        )[0]

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

    def save_npz(self, path) -> None:
        """Save the whole index into one ``.npz`` file (no pickle).

        The file holds the float32 unit vectors, ``cuis``, ``surfaces``,
        ``source_kbs`` and a JSON ``meta`` blob (format tag, count, dimension,
        ``encoder_info``).  The encoder **model weights are not embedded**: a
        reader must load the same checkpoint named in ``encoder_info`` to embed
        queries into this vector space.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "format": SINGLE_FILE_FORMAT,
            "count": len(self),
            "dim": int(self.vectors.shape[1]),
            "encoder_info": self.encoder_info,
        }
        # Unicode arrays keep the artifact pickle-free; None -> "" for source_kb.
        with path.open("wb") as handle:
            np.savez_compressed(
                handle,
                vectors=self.vectors,
                cuis=np.asarray(list(self.cuis), dtype=np.str_),
                surfaces=np.asarray(list(self.surfaces), dtype=np.str_),
                source_kbs=np.asarray(
                    ["" if kb is None else kb for kb in self.source_kbs],
                    dtype=np.str_,
                ),
                meta=np.asarray(json.dumps(meta, ensure_ascii=False)),
            )

    @classmethod
    def load_npz(cls, path) -> "VectorIndex":
        """Load an index written by :meth:`save_npz`."""
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            required = ("vectors", "cuis", "surfaces", "source_kbs", "meta")
            missing = [key for key in required if key not in data]
            if missing:
                raise ValueError(
                    f"{path}: npz artifact is missing keys: {', '.join(missing)}"
                )
            vectors = np.asarray(data["vectors"], dtype=np.float32)
            cuis = [str(value) for value in data["cuis"]]
            surfaces = [str(value) for value in data["surfaces"]]
            source_kbs = [str(value) or None for value in data["source_kbs"]]
            try:
                meta = json.loads(str(data["meta"]))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: invalid 'meta' JSON: {exc.msg}") from exc

        if not isinstance(meta, dict):
            raise ValueError(f"{path}: 'meta' must be a JSON object")
        if meta.get("format") != SINGLE_FILE_FORMAT:
            raise ValueError(
                f"{path}: unsupported artifact format {meta.get('format')!r}; "
                f"expected {SINGLE_FILE_FORMAT!r}"
            )
        if meta.get("count") != len(cuis) or meta.get("count") != vectors.shape[0]:
            raise ValueError(
                f"{path}: meta count {meta.get('count')!r} disagrees with "
                f"vectors/cuis ({vectors.shape[0]}/{len(cuis)})"
            )
        if meta.get("dim") != vectors.shape[1]:
            raise ValueError(
                f"{path}: meta dim {meta.get('dim')!r} disagrees with vectors "
                f"shape {vectors.shape!r}"
            )

        return cls(
            vectors,
            cuis,
            surfaces,
            source_kbs,
            encoder_info=meta.get("encoder_info"),
        )

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
