"""C3: text encoder used for mentions, feedback and KB surface forms.

`sentence_transformers` is imported lazily so that the rest of the package
(index, fusion, aggregation) works with numpy alone.
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from .query import mention_query


class Encoder(Protocol):
    @property
    def dim(self) -> int: ...

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Return an ``(n, dim)`` float32 array, L2-normalised per row.

        An empty input yields ``(0, dim)`` without invoking the model.
        """

    def encode_raw(self, texts: Sequence[str]) -> np.ndarray:
        """The same rows *before* that normalisation (model output as-is)."""


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
        # SentenceTransformer only calls ``self.eval()`` inside ``encode()``.
        # ``encode_mention_in_context`` drives ``auto_model`` directly and is
        # the first model call in the pipeline, so without this the mention
        # tokens are pooled with dropout active.
        self._model.eval()
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
        spans: Sequence[tuple[int, int] | None] | None = None,
    ) -> np.ndarray:
        """Embed a mention from its own token positions inside its context.

        Each context is tokenized as written with ``[SEP]`` markers around the
        mention (``common.query.mention_query``); the query is the mean of the
        last hidden states between the markers, i.e. of the mention tokens only
        and **nothing else**. Words before and after the mention stay context:
        they shape it through attention without being averaged into its vector.

        ``Q_0 = mean-pool( mention token states )`` — no ``E(m)`` term is added.
        The pooled vector is returned raw: the retrieval vector is L2-normalised
        exactly once, after ``Q_GRF`` is formed
        (``fusion.interpolate_matrix``), and normalising here would hand the
        fusion a unit vector and fix its weight against the feedback components
        at 1:1 regardless of their lengths.

        ``spans`` holds the gold ``(start_token, end_token)`` per row; ``None``
        falls back to locating the mention string in the sentence.

        Rows are **not** L2-normalised; see :meth:`encode_raw`.
        """
        mention_list = list(mentions)
        context_list = list(contexts)
        if len(mention_list) != len(context_list):
            raise ValueError(
                f"got {len(mention_list)} mentions but {len(context_list)} contexts"
            )
        if spans is None:
            span_list: list[tuple[int, int] | None] = [None] * len(mention_list)
        else:
            span_list = list(spans)
            if len(span_list) != len(mention_list):
                raise ValueError(
                    f"got {len(span_list)} spans for {len(mention_list)} mentions"
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

        texts: list[str] = []
        char_spans: list[tuple[int, int]] = []
        for mention, context, span in zip(mention_list, context_list, span_list):
            start_token, end_token = span if span is not None else (None, None)
            text, mention_start, mention_end = mention_query(
                mention,
                context,
                start_token=start_token,
                end_token=end_token,
            )
            texts.append(text)
            char_spans.append((mention_start, mention_end))
        out = np.zeros((len(texts), self.dim), dtype=np.float32)

        for start in range(0, len(texts), self._batch_size):
            texts_chunk = texts[start : start + self._batch_size]
            spans_chunk = char_spans[start : start + self._batch_size]
            encoded = tokenizer(
                texts_chunk,
                return_tensors="pt",
                padding=True,
                # Same cut-off ``encode()`` uses; without ``max_length`` the
                # tokenizer falls back to the checkpoint's declared limit and
                # the two paths could tokenise the same context differently.
                truncation=True,
                max_length=self._model.max_seq_length,
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

            for row, (mention_start, mention_end) in enumerate(spans_chunk):
                # Mention tokens: real tokens (end > 0) overlapping the mention
                # span between the two [SEP] markers. Special tokens have
                # offset (0, 0) and are excluded by ``end > 0``.
                keep = (
                    (offsets[row, :, 1] > 0)
                    & (offsets[row, :, 0] < mention_end)
                    & (offsets[row, :, 1] > mention_start)
                )
                positions = keep.nonzero(as_tuple=False).squeeze(-1)
                if positions.numel() == 0:
                    raise ValueError(
                        f"no tokens found inside mention {mention_list[start + row]!r}"
                    )
                # A mention past ``max_seq_length`` would otherwise be pooled
                # from a prefix of itself without any warning.
                if int(offsets[row, positions[-1], 1]) < mention_end:
                    raise ValueError(
                        f"mention {mention_list[start + row]!r} is truncated by "
                        f"max_seq_length={self._model.max_seq_length}"
                    )
                vector = hidden[row, positions.to(hidden.device)].mean(dim=0)
                out[start + row] = vector.float().cpu().numpy()

        if not np.all(np.isfinite(out)):
            raise ValueError("encoder output contains non-finite values")
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        if not np.all(np.isfinite(norms)) or np.any(norms <= 0.0):
            raise ValueError("encoder output contains a zero-norm vector")
        # Q_0 = mean-pool(mention token states in its sentence) and nothing else.
        # Returned raw: the one L2 normalisation of a retrieval vector happens
        # after Q_GRF is formed, inside the fusion.
        return out.astype(np.float32, copy=False)

    def encode_raw(self, texts: Sequence[str]) -> np.ndarray:
        """``E(·)`` exactly as the model returns it, no L2 normalisation.

        This is the ``E(m)`` / ``E(F_j)`` of the reference formula: the fusion
        adds and weights these rows as they are, so scaling one here would
        silently change its weight (architecture §4.4, reference §5 step 2).
        """
        if isinstance(texts, str):
            raise ValueError(
                "encode_raw() expects a sequence of strings, not a string"
            )
        items = list(texts)
        if not items:
            return np.zeros((0, self.dim), dtype=np.float32)

        for text in items:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("encode_raw() texts must be non-empty strings")

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

        return vectors.astype(np.float32, copy=False)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """``E(·)`` L2-normalised per row.

        Only the KB index rows and a stand-alone query go through this; the
        components of ``Q_GRF`` use :meth:`encode_raw`, because the retrieval
        vector is normalised exactly once, after the fusion sum.
        """
        vectors = self.encode_raw(texts)
        if vectors.shape[0] == 0:
            return vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return (vectors / norms).astype(np.float32, copy=False)
