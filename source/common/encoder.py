"""C3: text encoder used for mentions, feedback and KB surface forms.

`sentence_transformers` is imported lazily so that the rest of the package
(index, fusion, aggregation) works with numpy alone.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np


class Encoder(Protocol):
    @property
    def dim(self) -> int: ...

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return an ``(n, dim)`` float32 array, L2-normalised per row.

        An empty input yields ``(0, dim)`` without invoking the model.
        """


class SentenceTransformerEncoder:
    """Encoder backed by a SentenceTransformer checkpoint."""

    def __init__(
        self,
        model_name: str,
        *,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required by SentenceTransformerEncoder. "
                "Install it with `pip install sentence-transformers`."
            ) from exc

        kwargs = {} if device is None else {"device": device}
        self._model = SentenceTransformer(model_name, **kwargs)
        self._batch_size = batch_size
        self._model_name = model_name
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        """Embedding dimension; resolved lazily and cached."""
        if self._dim is None:
            declared = self._model.get_sentence_embedding_dimension()
            if declared is None:
                probe = np.asarray(
                    self._model.encode(
                        ["x"],
                        batch_size=1,
                        convert_to_numpy=True,
                        normalize_embeddings=False,
                        show_progress_bar=False,
                    ),
                    dtype=np.float32,
                )
                if probe.ndim != 2 or probe.shape[0] != 1:
                    raise RuntimeError(
                        "encoder produced an unexpected probe shape "
                        f"{probe.shape!r}"
                    )
                declared = probe.shape[1]
            declared = int(declared)
            if declared < 1:
                raise RuntimeError(f"encoder reported invalid dimension {declared}")
            self._dim = declared
        return self._dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if isinstance(texts, str):
            raise ValueError("encode() expects a sequence of strings, not a string")
        items = list(texts)
        if not items:
            return np.zeros((0, self.dim), dtype=np.float32)

        for text in items:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("encode() texts must be non-empty strings")

        raw = self._model.encode(
            items,
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        vectors = np.asarray(raw, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError(
                f"encoder output must be 2-D, got shape {vectors.shape!r}"
            )
        if vectors.shape[0] != len(items):
            raise ValueError(
                f"encoder returned {vectors.shape[0]} rows for {len(items)} texts"
            )
        if not np.all(np.isfinite(vectors)):
            raise ValueError("encoder output contains non-finite values")

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if not np.all(np.isfinite(norms)) or np.any(norms <= 0.0):
            raise ValueError("encoder output contains a zero-norm vector")

        dim = int(vectors.shape[1])
        if self._dim is None:
            self._dim = dim
        elif dim != self._dim:
            raise ValueError(
                f"encoder dimension changed from {self._dim} to {dim}"
            )

        return (vectors / norms).astype(np.float32, copy=False)
