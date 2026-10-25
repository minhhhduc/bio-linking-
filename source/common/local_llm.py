"""Minimal local causal-LM client (transformers only, greedy decoding).

Notebooks that need one short answer per item use this instead of pulling in a
pipeline: no API, no quantization, no prompt engineering. The checkpoint is a
Hugging Face repo id (downloaded on first use) or a local directory.
"""

from __future__ import annotations

import logging
from typing import Sequence

LOGGER = logging.getLogger(__name__)


class LocalLLMError(RuntimeError):
    """The model could not be loaded or produced no usable output."""


class LocalCausalLM:
    """``transformers`` + ``generate`` with ``do_sample=False``."""

    DTYPES = ("float16", "bfloat16", "float32")

    def __init__(
        self,
        model: str,
        *,
        max_new_tokens: int = 512,
        dtype: str = "bfloat16",
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise LocalLLMError("model must be a Hugging Face repo id or local path")
        if dtype not in self.DTYPES:
            raise LocalLLMError(
                f"unsupported dtype {dtype!r}; choose one of {self.DTYPES}"
            )
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            raise LocalLLMError("max_new_tokens must be an integer")
        if max_new_tokens < 1:
            raise LocalLLMError("max_new_tokens must be >= 1")
        try:
            import torch
            import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise LocalLLMError("requires torch and transformers") from exc

        self._torch = torch
        self._max_new_tokens = max_new_tokens
        self._tokenizer = AutoTokenizer.from_pretrained(
            model, trust_remote_code=False
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"

        kwargs = {
            "trust_remote_code": False,
            # Put the weights straight on the GPU when there is one.
            "device_map": "auto" if torch.cuda.is_available() else None,
        }
        # transformers >= 4.56 uses `dtype`; older releases use `torch_dtype`.
        major, minor = (int(part) for part in transformers.__version__.split(".")[:2])
        kwargs["dtype" if (major, minor) >= (4, 56) else "torch_dtype"] = getattr(
            torch, dtype
        )
        self._model = AutoModelForCausalLM.from_pretrained(model, **kwargs)
        self._model.eval()
        self.device = self._model.device

    def _template(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        template = getattr(self._tokenizer, "chat_template", None)
        if not template:
            return prompt
        try:
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            if "enable_thinking" in template:
                raise LocalLLMError(
                    "chat template supports enable_thinking but tokenizer rejected it"
                )
            LOGGER.warning("chat template has no enable_thinking flag")
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    def complete_batch(self, prompts: Sequence[str]) -> list[str]:
        if isinstance(prompts, str):
            raise LocalLLMError("complete_batch() expects a sequence of prompts")
        items = [self._template(prompt) for prompt in prompts]
        if not items:
            return []
        encoded = self._tokenizer(
            items, return_tensors="pt", padding=True, add_special_tokens=False
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        input_length = encoded["input_ids"].shape[1]
        with self._torch.inference_mode():
            outputs = self._model.generate(
                **encoded,
                do_sample=False,  # temperature = 0
                max_new_tokens=self._max_new_tokens,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        return self._tokenizer.batch_decode(
            outputs[:, input_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def complete(self, prompt: str) -> str:
        return self.complete_batch([prompt])[0]
