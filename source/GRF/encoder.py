"""C3 encoder, re-exported from :mod:`common.encoder`."""

from __future__ import annotations

from common.encoder import Encoder, SentenceTransformerEncoder

from .config import PipelineConfig


def build_encoder(config: PipelineConfig) -> SentenceTransformerEncoder:
    """Build the encoder named by ``config.encoder`` (a HuggingFace id)."""
    return SentenceTransformerEncoder(config.encoder)


__all__ = ["Encoder", "SentenceTransformerEncoder", "build_encoder"]
