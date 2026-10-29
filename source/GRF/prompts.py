"""Prompt templates for C5 feedback and C8 re-ranking, read from ``prompts.yaml``.

The text lives next to this module so it can be edited without touching code.
This module only loads and checks it.

The cache namespace uses ``digest()``, a hash of the file bytes, so editing a
template automatically invalidates the feedback that template produced.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

PROMPT_FILE = Path(__file__).with_name("prompts.yaml")

# Placeholders each template must carry.  A template missing one would send the
# LLM a prompt that never names the entity, which fails silently at run time.
_REQUIRED = {
    ("feedback", "envelope"): {"context", "mention", "instruction"},
    ("feedback", "instructions", "def"): {"m"},
    ("feedback", "instructions", "syn"): {"m", "n"},
    ("feedback", "instructions", "std"): {"m", "KB"},
    ("rerank", "candidate_row"): {"index", "cui", "name", "surface"},
    ("rerank", "instruction"): {"mention", "context", "candidates"},
}


def load(path: Path = PROMPT_FILE) -> dict:
    """Parse ``prompts.yaml`` and assert every template has its placeholders."""
    import string

    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "Reading the prompt templates needs PyYAML; install it with "
            "`pip install pyyaml>=6.0`."
        ) from exc

    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"{path.name} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path.name} must be a mapping at the top level")

    for keys, required in _REQUIRED.items():
        node = data
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                raise ValueError(f"{path.name}: missing {'/'.join(keys)}")
            node = node[key]
        if not isinstance(node, str) or not node.strip():
            raise ValueError(f"{path.name}: {'/'.join(keys)} must be a string")
        try:
            found = {
                name for _, name, _, _ in string.Formatter().parse(node)
                if name is not None
            }
        except ValueError as exc:
            raise ValueError(f"{path.name}: {'/'.join(keys)}: {exc}") from exc
        if missing := required - found:
            raise ValueError(
                f"{path.name}: {'/'.join(keys)} is missing {sorted(missing)}"
            )
    return data


def digest(path: Path = PROMPT_FILE) -> str:
    """Short hash of the prompt file, for the feedback cache namespace."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


_PROMPTS: dict | None = None


def get_prompts(path: Path = PROMPT_FILE) -> dict:
    """The templates, loaded once per process."""
    global _PROMPTS
    if _PROMPTS is None or path != PROMPT_FILE:
        _PROMPTS = load(path)
    return _PROMPTS
