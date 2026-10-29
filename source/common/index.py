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

# A row whose norm is this close to 1 is already unit-length: rescaling it
# would be a second normalisation of the same vector, not a rounding fix.
_UNIT_TOLERANCE = 1e-6


def as_unit_rows(matrix: np.ndarray) -> np.ndarray:
    """Scale rows to unit length, leaving rows that already are unit *as given*.

    A cosine scan wants unit rows on both sides, and the index normalises what
    it is handed.  The query it is handed is normally ``Q_GRF``, which the
    fusion already normalised after the weighted sum — dividing that again
    would be the second normalisation of the same retrieval vector.  Rows that
    are not unit-length (a raw component, a caller's unnormalised query) are
    scaled once here, which is the only normalisation they get.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    needs_scale = (np.abs(norms - 1.0) > _UNIT_TOLERANCE).ravel()
    if not np.any(needs_scale):
        return matrix
    scaled = np.array(matrix, dtype=matrix.dtype, copy=True)
    scaled[needs_scale] = (
        matrix[needs_scale] / norms[needs_scale]
    )
    return scaled


class VectorIndex:
    """One entry per KB surface form; exact cosine scan at query time."""

    # Bounds the transient (N, queries) score matrix on the GPU.
    # Queries per device matmul, capped so the chunk's score matrix stays
    # around 256 MiB of float32 whatever the index size (T4-friendly).
    _GPU_QUERY_CHUNK = 256
    _GPU_SCORE_BUDGET = 1 << 26
    # Same idea for the CPU path: floats of the (rows, N) score block.
    _CPU_SCORE_BUDGET = 1 << 25

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
            # Rows that arrive unit-length (``encoder.encode`` normalises them
            # once) are stored as given; only a raw row is scaled, and then
            # exactly once.
            array = as_unit_rows(array)

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

        expected_dim: int | None = None
        output: np.ndarray | None = None
        done = 0
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
                # Preallocate once so a multi-million-row KB is not held twice.
                output = np.empty(
                    (len(surfaces), expected_dim), dtype=np.float32
                )
            elif encoded.shape[1] != expected_dim:
                raise ValueError("encoder returned inconsistent dimensions")
            output[done : done + len(chunk)] = encoded
            done += len(chunk)
            if progress_callback is not None:
                progress_callback(done, len(surfaces))

        assert output is not None  # entries is non-empty, so the loop ran once
        return cls(
            output,
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
        matrix: np.ndarray,
        top_k: int,
        source_kb: str | None,
    ) -> list[list[RetrievalHit]]:
        """Exact cosine scan on CPU: one matmul per query chunk, no row loop.

        The score block is ``S = Q · Aᵀ`` (both sides unit-norm).  The k-th
        score of every row is found with a partition **on a copy**, so the
        score block keeps its original column order and a column index still
        addresses the same index row.  Every entry at or above that k-th score
        is kept, so ties at the k-th score are resolved by the documented rule
        ``(-score, cui, surface)`` instead of by whichever entry the partition
        happened to leave inside the slice.
        """
        count = len(self)
        empty: list[list[RetrievalHit]] = [[] for _ in range(matrix.shape[0])]
        if count == 0:
            return empty
        # Rows are validated non-zero in search_batch.  Already-unit rows (the
        # normal case: the fusion normalised Q_GRF once) are used unchanged, so
        # this is not a second normalisation of the same vector.
        matrix = as_unit_rows(matrix)

        # Rows of the index that are eligible for this search.  When a
        # source_kb filter is given, the eligible rows form a compact matrix,
        # so the k-th largest is looked up in the *candidate* block, not in a
        # full-width block padded with -inf.
        if source_kb is None:
            positions = np.arange(count, dtype=np.int64)
        else:
            positions = np.fromiter(
                (
                    index
                    for index, name in enumerate(self.source_kbs)
                    if name == source_kb
                ),
                dtype=np.int64,
            )
        candidates = int(positions.size)
        if candidates == 0:
            return empty
        columns = min(top_k, candidates)
        chunk_rows = max(
            1, min(self._GPU_QUERY_CHUNK, self._CPU_SCORE_BUDGET // count)
        )
        results: list[list[RetrievalHit]] = []
        for start in range(0, matrix.shape[0], chunk_rows):
            chunk = matrix[start : start + chunk_rows]
            # One matrix multiply over the whole index, as the contract says.
            scores = chunk @ self.vectors.T
            # The source_kb filter only selects columns of that same score
            # block, so a filtered and an unfiltered search score the same
            # row identically.
            block = scores if source_kb is None else scores[:, positions]
            # np.partition returns a copy; the k-th largest of each row lands
            # at index candidates - columns of that copy.
            cutoff = np.partition(block, candidates - columns, axis=1)[
                :, candidates - columns
            ]
            rows, cols = np.nonzero(block >= cutoff[:, None])
            if rows.size == 0:
                results.extend([[] for _ in range(chunk.shape[0])])
                continue
            values = block[rows, cols]
            results.extend(
                self._assemble(
                    rows,
                    positions[cols],
                    values,
                    columns,
                    chunk.shape[0],
                )
            )
        return results

    def _assemble(
        self,
        columns_of_row: np.ndarray,
        indices: np.ndarray,
        values: np.ndarray,
        limit: int,
        n_queries: int,
    ) -> list[list[RetrievalHit]]:
        """Order the candidate block and cut it to ``limit`` per query row.

        ``columns_of_row`` is the query row of every candidate, ``indices`` the
        index row it came from and ``values`` its score.  One ``np.lexsort``
        orders every query row at once by ``(-score, cui, surface)`` — the
        query row is the primary key, the tie-break keys come last because
        ``np.lexsort`` gives priority to the *last* key.
        """
        flat = indices.astype(np.int64, copy=False)
        cuis = np.asarray([self.cuis[i] for i in flat], dtype=np.str_)
        surfaces = np.asarray([self.surfaces[i] for i in flat], dtype=np.str_)
        order = np.lexsort(
            (surfaces, cuis, -np.asarray(values, dtype=np.float64), columns_of_row)
        )
        rows = columns_of_row[order]
        cols = flat[order]
        values = np.asarray(values, dtype=np.float64)[order]

        # Keep the first ``limit`` candidates of every query row.
        widths = np.bincount(rows, minlength=n_queries)
        offsets = np.concatenate(([0], np.cumsum(widths)))
        position = np.arange(rows.size) - offsets[rows]
        kept = np.flatnonzero(position < limit)
        kept_rows = rows[kept]
        kept_widths = np.bincount(kept_rows, minlength=n_queries)
        kept_offsets = np.concatenate(([0], np.cumsum(kept_widths)))

        hits: list[list[RetrievalHit]] = []
        for row in range(n_queries):
            base = kept[kept_offsets[row] : kept_offsets[row + 1]]
            hits.append(
                [
                    RetrievalHit(
                        cui=self.cuis[int(cols[i])],
                        surface=self.surfaces[int(cols[i])],
                        score=float(values[i]),
                        rank=rank,
                    )
                    for rank, i in enumerate(base.tolist(), start=1)
                ]
            )
        return hits

    def search_batch(
        self,
        queries,
        top_k: int,
        *,
        source_kb: str | None = None,
    ) -> list[list[RetrievalHit]]:
        """Top-k exact cosine hits for every row of ``queries``.

        Scores are one matrix product, ``R = queries @ self.vectors.T`` (both
        sides are unit-norm, so the product is cosine), then top-k along each
        row; the batch is chunked so the ``(rows, N)`` block stays bounded.
        The ordering rule is identical to calling :meth:`search` per row.  With
        :meth:`use_gpu` active, the product and ``topk`` run on the device in
        query chunks and only rows at or above each k-th score are copied back
        to CPU; float32 scores can differ in the last ULPs from the CPU path.

        Query rows that are already unit-length are scored **as given**: the
        fusion normalises ``Q_GRF`` once after the weighted sum, and scaling it
        again here would be a second normalisation of the same vector.
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
        matrix = as_unit_rows(matrix)
        if self._device is not None:
            return self._search_gpu(matrix, top_k, source_kb)
        return self._search_cpu(matrix, top_k, source_kb)

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

        chunk_rows = max(
            1, min(self._GPU_QUERY_CHUNK, self._GPU_SCORE_BUDGET // candidate_count)
        )
        results: list[list[RetrievalHit]] = []
        for start in range(0, matrix.shape[0], chunk_rows):
            chunk = matrix[start : start + chunk_rows]
            with torch.no_grad():
                queries = torch.from_numpy(
                    np.ascontiguousarray(chunk)
                ).to(device)
                # ``search_batch`` already made every row unit-length (and left
                # the ones that arrived unit untouched), so no rescaling here:
                # a unit row's dot product with a unit index row is the cosine.
                scores = self._gpu_vectors @ queries.T
                if rows is not None:
                    scores = scores.index_select(0, rows)

                # top-k per query column on the device, then one vectorised
                # tie-break (score desc, cui, surface) on the k hits of every
                # column. Scores can differ in the last float32 ULPs from BLAS,
                # but the ordering rule is the same as the CPU path.
                values, indices = torch.topk(scores, k=columns, dim=0)
                values = values.cpu().numpy()
                indices = indices.cpu().numpy()
                if positions_cpu is not None:
                    indices = positions_cpu[indices]
                ordered_indices, ordered_values = self._order_topk(indices, values)
                results.extend(self._hits(ordered_indices, ordered_values))
        return results

    def _order_topk(
        self, indices: np.ndarray, values: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sort a ``(k, rows)`` hit block by ``(-score, cui, surface)``.

        ``np.lexsort(..., axis=0)`` sorts every column independently, so the
        whole chunk is ordered in one call instead of one Python ``sorted`` per
        mention. ``cui`` / ``surface`` are compared as strings, matching the
        rule documented on :meth:`search`.
        """
        flat = indices.ravel().astype(np.int64, copy=False)
        cuis = np.asarray(
            [self.cuis[i] for i in flat], dtype=np.str_
        ).reshape(indices.shape)
        surfaces = np.asarray(
            [self.surfaces[i] for i in flat], dtype=np.str_
        ).reshape(indices.shape)
        order = np.lexsort(
            (surfaces, cuis, -np.asarray(values, dtype=np.float64)), axis=0
        )
        return (
            np.take_along_axis(indices, order, axis=0),
            np.take_along_axis(values, order, axis=0),
        )

    def _hits(
        self, indices: np.ndarray, values: np.ndarray
    ) -> list[list[RetrievalHit]]:
        """``(k, rows)`` sorted hit arrays -> one hit list per query row."""
        return [
            [
                RetrievalHit(
                    cui=self.cuis[int(indices[rank, column])],
                    surface=self.surfaces[int(indices[rank, column])],
                    score=float(values[rank, column]),
                    rank=rank + 1,
                )
                for rank in range(indices.shape[0])
            ]
            for column in range(indices.shape[1])
        ]

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
