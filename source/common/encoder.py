"""C3: text encoder used for mentions, feedback and KB surface forms.

`sentence_transformers` is imported lazily so that the rest of the package
(index, fusion, aggregation) works with numpy alone.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from .query import query_text


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

    def encode_mention_in_context(
        self,
        mentions: Sequence[str],
        contexts: Sequence[str],
    ) -> np.ndarray:
        """Embed a mention from its own token positions inside its context.

        Each input is tokenized as ``mention [SEP] context`` and run through the
        transformer; the returned vector is the mean of the last hidden states of
        the *mention* tokens only, so the context shapes the mention through
        attention without being averaged into its vector. The mention is the
        prefix, so it survives tokenizer truncation of long contexts.

        Rows are L2-normalised, like :meth:`encode`.
        """
        mention_list = list(mentions)
        context_list = list(contexts)
        if len(mention_list) != len(context_list):
            raise ValueError(
                f"got {len(mention_list)} mentions but {len(context_list)} contexts"
            )
        if not mention_list:
            return np.zeros((0, self.dim), dtype=np.float32)

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("encode_mention_in_context requires torch") from exc

        module = self._model[0]
        tokenizer = getattr(module, "tokenizer", None)
        auto_model = getattr(module, "auto_model", None)
        if tokenizer is None or auto_model is None:
            raise RuntimeError(
                "the checkpoint has no transformer module to pool token states from"
            )
        if not getattr(tokenizer, "is_fast", False):
            raise RuntimeError(
                "encode_mention_in_context needs a fast tokenizer for character "
                "offsets; use a BERT-style checkpoint such as SapBERT"
            )
        device = self._model.device

        texts = [
            query_text(mention, context)
            for mention, context in zip(mention_list, context_list)
        ]
        mention_ends = [len(mention) for mention in mention_list]
        out = np.zeros((len(texts), self.dim), dtype=np.float32)

        for start in range(0, len(texts), self._batch_size):
            texts_chunk = texts[start : start + self._batch_size]
            ends_chunk = mention_ends[start : start + self._batch_size]
            encoded = tokenizer(
                texts_chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                return_offsets_mapping=True,
            )
            offsets = encoded.pop("offset_mapping")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                hidden = auto_model(**encoded).last_hidden_state
            if hidden.shape[-1] != out.shape[1]:
                raise ValueError(
                    f"token states are {hidden.shape[-1]}-d but the encoder "
                    f"reports {out.shape[1]}-d"
                )

            for row, mention_end in enumerate(ends_chunk):
                # Mention tokens: real tokens (end > 0) that end inside the
                # mention prefix. Special tokens have offset (0, 0).
                keep = (offsets[row, :, 1] > 0) & (offsets[row, :, 1] <= mention_end)
                positions = keep.nonzero(as_tuple=False).squeeze(-1)
                if positions.numel() == 0:
                    raise ValueError(
                        f"no tokens found inside mention {mention_list[start + row]!r}"
                    )
                vector = hidden[row, positions.to(hidden.device)].mean(dim=0)
                out[start + row] = vector.float().cpu().numpy()

        if not np.all(np.isfinite(out)):
            raise ValueError("encoder output contains non-finite values")
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        if not np.all(np.isfinite(norms)) or np.any(norms <= 0.0):
            raise ValueError("encoder output contains a zero-norm vector")
        return (out / norms).astype(np.float32, copy=False)

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
