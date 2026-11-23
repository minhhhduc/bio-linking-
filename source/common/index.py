"""C4: exact cosine vector index over KB surface forms.

Vectors are stored L2-normalised as float32, so a dot product is cosine
similarity.  The index is a ``pycauset`` matrix: persisted as a single
``.npz`` file (:meth:`VectorIndex.save_npz`) but **loaded** by streaming its
``vectors.npy`` member to local disk and importing it under a RAM ceiling
(:meth:`VectorIndex.load_npz`), so the payload lives in a backing file and the
KB is never held whole in RAM.  Scoring is ``pycauset.matmul`` over that
disk-backed matrix; top-k stays in NumPy
(``pycauset`` has no ``topk``).  A directory form (``vectors.npy`` +
``meta.json``, :meth:`VectorIndex.save`/:meth:`VectorIndex.load`) is still
supported; neither form uses pickle.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pycauset as pc

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


def _extract_vectors(path: Path, cache_dir) -> Path:
    """Stream the ``vectors.npy`` member of an ``.npz`` artifact to local disk.

    ``np.load`` on an ``.npz`` decompresses members in RAM; ``pycauset`` can
    only memory-map a plain ``.npy``.  Copying the zip member block by block
    gives that file without ever holding the whole array in the heap.  The
    directory is keyed by the artifact's size and mtime, so rebuilding the KB
    re-extracts instead of silently scoring against stale vectors.
    """
    stat = path.stat()
    root = (
        Path(cache_dir)
        if cache_dir is not None
        else Path(tempfile.gettempdir()) / "grf_kb_npy"
    )
    target = root / f"{path.stem}-{stat.st_size}-{stat.st_mtime_ns}" / "vectors.npy"
    if target.exists():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    # A unique staging name: two notebooks may load the same artifact at once.
    staging = target.with_name(f"vectors.npy.{os.getpid()}.tmp")
    with zipfile.ZipFile(path) as archive:
        if "vectors.npy" not in archive.namelist():
            raise ValueError(f"{path}: npz artifact has no 'vectors.npy' member")
        with archive.open("vectors.npy") as source, staging.open("wb") as sink:
            shutil.copyfileobj(source, sink, length=1 << 20)
    staging.replace(target)
    return target


# Importing a ``.npy`` into ``pycauset`` copies it into native storage.  Above
# this ceiling that copy is a backing file on disk, not RAM, so a KB larger
# than free memory still loads; below it the RAM copy is cheap and faster.
_IMPORT_RAM_CEILING = 1 << 29


def _open_vectors(path: Path, backing_dir=None):
    """Open an extracted ``vectors.npy`` as a disk-backed matrix.

    ``np.load(mmap_mode="r")`` never materialises the array, and the RAM
    ceiling makes ``pycauset`` write its copy to a backing file instead of
    refusing the import when the array is larger than available memory.
    """
    if backing_dir is not None:
        pc.set_backing_dir(backing_dir)
    return pc.matrix(
        np.load(path, mmap_mode="r"), max_in_ram_bytes=_IMPORT_RAM_CEILING
    )


class VectorIndex:
    """One entry per KB surface form; exact cosine scan at query time."""

    # Queries per matmul, capped so the chunk's (rows, N) score block stays
    # around 256 MiB of float32 whatever the index size (T4-friendly).
    _QUERY_CHUNK = 256
    _SCORE_BUDGET = 1 << 26

    def __init__(
        self,
        vectors,
        cuis: Sequence[str],
        surfaces: Sequence[str],
        source_kbs: Sequence[str | None] | None = None,
        *,
        encoder_info: Mapping[str, str] | None = None,
    ) -> None:
        # A loaded artifact arrives as a disk-backed ``pycauset`` matrix: it is
        # already L2-normalised (the writer normalised it once), so it must not
        # be copied through a NumPy array — that would put the whole KB in RAM.
        if isinstance(vectors, np.ndarray):
            array = np.array(vectors, dtype=np.float32, copy=True)
        else:
            array = vectors
        if len(array.shape) != 2:
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
        count = int(array.shape[0])
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

        if isinstance(array, np.ndarray):
            if not np.all(np.isfinite(array)):
                raise ValueError("vectors contain non-finite values")
            if count:
                norms = np.linalg.norm(array, axis=1, keepdims=True)
                if not np.all(np.isfinite(norms)) or np.any(norms <= 0.0):
                    raise ValueError("vectors contain a zero-norm row")
                # Rows that arrive unit-length (``encoder.encode`` normalises
                # them once) are stored as given; only a raw row is scaled, and
                # then exactly once.
                array = as_unit_rows(array)
            # One copy into ``pycauset`` storage so the scan is a single
            # ``pycauset.matmul`` against a matrix the backend owns.
            self.vectors = pc.matrix(array) if count else array
        else:
            self.vectors = array

        self.cuis = cuis
        self.surfaces = surfaces
        self.source_kbs = source_kbs
        self.encoder_info = encoder_info
        self._device: str | None = None
        self._gpu_backend: str | None = None
        # The extracted ``vectors.npy`` behind :attr:`vectors`, when the index
        # was loaded from an artifact.  The GPU scan streams row blocks from
        # this file; pycauset matrices cannot be sliced into operands.
        self._npy_path: Path | None = None

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
        """``"cuda"`` when the scan runs on the GPU, else ``"cpu"``."""
        return self._device or "cpu"

    def use_gpu(self, device: str = "cuda") -> None:
        """Score on the GPU with CuPy.

        ``cupy`` runs ``Q · A_blockᵀ`` on index row blocks copied to VRAM;
        the KB stays memory-mapped on disk and is never copied whole.
        """
        if device == "cpu":
            self.release_gpu()
            return
        if device != "cuda":
            raise ValueError("device must be 'cuda' or 'cpu'")
        if self._device == device:
            return
        try:
            import cupy as cp
        except ImportError as exc:
            raise RuntimeError(
                "device='cuda' needs CuPy; install the build for your CUDA "
                "runtime (e.g. pip install cupy-cuda12x)"
            ) from exc
        try:
            cp.cuda.runtime.getDeviceCount()
        except Exception as exc:  # noqa: BLE001 - CuPy raises its own error type
            raise RuntimeError(
                "device='cuda' but CuPy sees no CUDA device; use the CPU path"
            ) from exc
        cp.get_default_memory_pool().free_all_blocks()
        self._device = device
        self._gpu_backend = "cupy"

    def release_gpu(self) -> None:
        """Stop scoring on the GPU and release device memory."""
        self._device = None
        self._gpu_backend = None
        try:
            import cupy as cp
        except ImportError:
            return
        cp.get_default_memory_pool().free_all_blocks()

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

    def _search(
        self,
        matrix: np.ndarray,
        top_k: int,
        source_kb: str | None,
    ) -> list[list[RetrievalHit]]:
        """Exact cosine scan: one ``pycauset.matmul`` per query chunk.

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
        if self._device == "cuda":
            rows, cols, values = self._cupy_candidates(matrix, columns, positions)
            if rows.size == 0:
                return empty
            return self._assemble(rows, cols, values, columns, matrix.shape[0])
        chunk_rows = max(
            1, min(self._QUERY_CHUNK, self._SCORE_BUDGET // count)
        )
        results: list[list[RetrievalHit]] = []
        for start in range(0, matrix.shape[0], chunk_rows):
            chunk = matrix[start : start + chunk_rows]
            # One matrix multiply over the whole index, as the contract says.
            # ``index.vectors.T`` is a metadata-only view of the memory-mapped
            # KB; the (rows, N) result is a bounded in-RAM block.
            scores = pc.to_numpy(
                pc.matmul(pc.matrix(chunk), self.vectors.T),
                allow_huge=True,
            )
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

    def _cupy_candidates(
        self,
        matrix: np.ndarray,
        columns: int,
        positions: np.ndarray,
    ):
        """GPU scan: index row blocks → VRAM → ``Q·Aᵀ`` with CuPy.

        Block width follows the free VRAM CuPy reports, so neither the KB nor
        the full score matrix is ever resident.  Returns the entries at or
        above each block's k-th score, in global index coordinates.
        """
        import cupy as cp

        if self._npy_path is not None:
            source = np.load(self._npy_path, mmap_mode="r")
        elif isinstance(self.vectors, np.ndarray):
            source = self.vectors
        else:
            source = pc.to_numpy(self.vectors, allow_huge=True)
        n_rows = int(source.shape[0])
        dim = int(source.shape[1])
        n_queries = int(matrix.shape[0])
        itemsize = np.dtype(np.float32).itemsize
        # A block costs its vectors, one score column per query, and the mask.
        free, _ = cp.cuda.Device().mem_info
        per_row = dim * itemsize + n_queries * (itemsize + 1)
        block_rows = max(1, int(free) // 3 // per_row)
        filtered = int(positions.size) != n_rows

        queries = cp.asarray(np.ascontiguousarray(matrix, dtype=np.float32))
        rows_out: list[np.ndarray] = []
        cols_out: list[np.ndarray] = []
        values_out: list[np.ndarray] = []
        try:
            for start in range(0, n_rows, block_rows):
                stop = min(n_rows, start + block_rows)
                lo = int(np.searchsorted(positions, start, side="left"))
                hi = int(np.searchsorted(positions, stop, side="left"))
                if hi <= lo:
                    continue
                scores = queries @ cp.asarray(source[start:stop]).T
                if filtered:
                    scores = scores[:, cp.asarray(positions[lo:hi] - start)]
                width = hi - lo
                kth = min(columns, width)
                # cp.partition copies, so the block keeps its column order.
                cutoff = cp.partition(scores, width - kth, axis=1)[
                    :, width - kth
                ]
                rows, cols = cp.nonzero(scores >= cutoff[:, None])
                rows_out.append(cp.asnumpy(rows))
                cols_out.append(positions[lo:hi][cp.asnumpy(cols)])
                values_out.append(cp.asnumpy(scores[rows, cols]))
                del scores
        finally:
            del queries
            cp.get_default_memory_pool().free_all_blocks()
        if not rows_out:
            empty = np.empty(0, dtype=np.int64)
            return empty, empty.copy(), np.empty(0, dtype=np.float32)
        return (
            np.concatenate(rows_out),
            np.concatenate(cols_out),
            np.concatenate(values_out),
        )

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
        The product is ``pycauset.matmul`` over the memory-mapped index; top-k
        runs in NumPy on that block.  The ordering rule is identical to
        calling :meth:`search` per row; float32 scores can differ in the last
        ULPs from BLAS.

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
        return self._search(matrix, top_k, source_kb)

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

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _as_numpy_vectors(self) -> np.ndarray:
        """The vectors as a NumPy array; materialises a disk-backed KB."""
        if isinstance(self.vectors, np.ndarray):
            return self.vectors
        return pc.to_numpy(self.vectors, allow_huge=True)

    def save(self, directory) -> None:
        base = Path(directory)
        base.mkdir(parents=True, exist_ok=True)

        pc.save_npy(self.vectors, base / "vectors.npy")
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
                vectors=self._as_numpy_vectors(),
                cuis=np.asarray(list(self.cuis), dtype=np.str_),
                surfaces=np.asarray(list(self.surfaces), dtype=np.str_),
                source_kbs=np.asarray(
                    ["" if kb is None else kb for kb in self.source_kbs],
                    dtype=np.str_,
                ),
                meta=np.asarray(json.dumps(meta, ensure_ascii=False)),
            )

    @classmethod
    def load_npz(cls, path, *, cache_dir=None) -> "VectorIndex":
        """Load an index written by :meth:`save_npz`.

        The ``vectors.npy`` member is streamed to ``cache_dir`` (default:
        ``tempfile.gettempdir()/grf_kb_npy``) and imported under a RAM
        ceiling, so a KB larger than free memory lands in a backing file
        instead of the Python heap.
        """
        path = Path(path)
        with np.load(path, allow_pickle=False) as data:
            required = ("vectors", "cuis", "surfaces", "source_kbs", "meta")
            missing = [key for key in required if key not in data]
            if missing:
                raise ValueError(
                    f"{path}: npz artifact is missing keys: {', '.join(missing)}"
                )
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
            if meta.get("count") != len(cuis):
                raise ValueError(
                    f"{path}: meta count {meta.get('count')!r} disagrees with "
                    f"cuis ({len(cuis)})"
                )

        extracted = _extract_vectors(path, cache_dir)
        vectors = _open_vectors(extracted, cache_dir)
        if vectors.shape[0] != len(cuis):
            raise ValueError(
                f"{path}: meta count {meta.get('count')!r} disagrees with "
                f"vectors ({vectors.shape[0]})"
            )
        if meta.get("dim") != vectors.shape[1]:
            raise ValueError(
                f"{path}: meta dim {meta.get('dim')!r} disagrees with vectors "
                f"shape {vectors.shape!r}"
            )

        index = cls(
            vectors,
            cuis,
            surfaces,
            source_kbs,
            encoder_info=meta.get("encoder_info"),
        )
        index._npy_path = extracted
        return index

    @classmethod
    def load(cls, directory) -> "VectorIndex":
        base = Path(directory)
        vectors = _open_vectors(base / "vectors.npy")
        with (base / "meta.json").open("r", encoding="utf-8-sig") as handle:
            meta = json.load(handle)

        if not isinstance(meta, dict):
            raise ValueError("meta.json must contain a JSON object")
        required = ("count", "dim", "cuis", "surfaces", "source_kbs")
        missing = [key for key in required if key not in meta]
        if missing:
            raise ValueError(f"meta.json is missing keys: {', '.join(missing)}")

        if len(vectors.shape) != 2:
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

        index = cls(
            vectors,
            cuis,
            surfaces,
            source_kbs,
            encoder_info=meta.get("encoder_info"),
        )
        index._npy_path = base / "vectors.npy"
        return index
