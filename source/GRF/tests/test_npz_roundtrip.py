"""Runnable check for the single-file KB artifact used by the Colab notebooks.

Run from the repository root: ``python source/GRF/tests/test_npz_roundtrip.py``.
numpy only; no encoder, no network.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from GRF.index import VectorIndex  # noqa: E402


def demo() -> None:
    index = VectorIndex(
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        ["C1", "C2"],
        ["alpha", "beta"],
        [None, "KB"],
        encoder_info={"model": "test-encoder"},
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "kb.npz"
        index.save_npz(path)
        assert path.is_file()

        loaded = VectorIndex.load_npz(path)
        assert list(loaded.cuis) == ["C1", "C2"]
        assert list(loaded.surfaces) == ["alpha", "beta"]
        assert list(loaded.source_kbs) == [None, "KB"]
        assert loaded.encoder_info == {"model": "test-encoder"}
        assert np.allclose(loaded.vectors, index.vectors)

        hits = loaded.search(np.array([1.0, 0.0], dtype=np.float32), 1)
        assert hits and hits[0].cui == "C1", hits

    print("npz roundtrip ok")


if __name__ == "__main__":
    demo()
