"""Local HuggingFace chat client for feedback (C5) and re-ranking (C8).

Implements the same :class:`~GRF.llm.LLMClient` contract as
:class:`~GRF.llm.OpenAICompatibleClient`, so ``FeedbackGenerator`` and
``LLMReranker`` use a locally loaded model without any other change.  The
weights come from ``transformers`` (HuggingFace Hub id or a local directory);
``torch`` and ``transformers`` are imported lazily, so importing this module
costs nothing and a missing dependency only surfaces as an ``LLMError`` that
degrades to the baseline path like any other failed LLM call.
"""

from __future__ import annotations

import gc
import math
import threading

from .llm import LLMError

DTYPES = ("auto", "float16", "bfloat16", "float32")


def _check_bool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise LLMError(f"{name} must be a boolean, got {value!r}")
    return value


def _check_positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise LLMError(f"{name} must be a positive integer, got {value!r}")
    return value


class TransformersChatClient:
    """Generate with a causal LM loaded through ``transformers``.

    ``model_name`` is a HuggingFace id or a local directory (for example a
    folder on Drive).  The model is loaded on the first :meth:`complete` call
    and stays resident until :meth:`unload`; generation is serialised with a
    lock because the callers issue concurrent feedback requests and a single
    resident model cannot serve several generations at once.

    ``device="auto"`` picks CUDA when the runtime has one.  ``dtype="auto"`` is
    then resolved from that resolved device (``bfloat16`` on CUDA, ``float32``
    on CPU); pass an explicit dtype to override it.  Note that a checkpoint's
    own config dtype (often ``float32``) is *not* what "auto" means here — use
    ``"float16"``/``"bfloat16"`` for a half-precision copy regardless of device.

    ``temperature=0.0`` (the default) is greedy decoding, matching the
    deterministic OpenAI path.  ``timeout_s`` is accepted for interface
    compatibility and ignored: a local generation cannot be interrupted
    mid-step without leaving the CUDA context in an undefined state.

    Any failure — a missing dependency, an unloadable checkpoint, a device the
    runtime does not have, an out-of-memory generation — surfaces as an
    ``LLMError``, exactly like a failed API call.
    """

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        trust_remote_code: bool = False,
        use_chat_template: bool = True,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise LLMError("model_name must be a non-empty string or path")
        self.model_name = model_name.strip()

        if not isinstance(device, str) or not device.strip():
            raise LLMError("device must be 'auto', 'cuda', 'cpu' or 'cuda:N'")
        self.device = device.strip()
        if self.device not in ("auto", "cpu") and not self.device.startswith("cuda"):
            raise LLMError(
                f"device must be 'auto', 'cpu' or start with 'cuda', got {device!r}"
            )

        if not isinstance(dtype, str) or dtype not in DTYPES:
            raise LLMError(f"dtype must be one of {DTYPES}, got {dtype!r}")
        self.dtype = dtype

        self.max_new_tokens = _check_positive_int("max_new_tokens", max_new_tokens)

        if isinstance(temperature, bool):
            raise LLMError(f"temperature must be a number, got {temperature!r}")
        try:
            temperature_value = float(temperature)
        except (TypeError, ValueError) as exc:
            raise LLMError(f"temperature must be a number, got {temperature!r}") from exc
        if not math.isfinite(temperature_value) or temperature_value < 0.0:
            raise LLMError(
                f"temperature must be a finite non-negative number, got {temperature!r}"
            )
        self.temperature = temperature_value

        self._trust_remote_code = _check_bool("trust_remote_code", trust_remote_code)
        self._use_chat_template = _check_bool("use_chat_template", use_chat_template)

        self._model = None
        self._tokenizer = None
        self._resolved_device: str | None = None
        # Guards loading and generation: one resident model, one generation.
        self._lock = threading.RLock()

    @property
    def loaded_device(self) -> str | None:
        """Device the weights live on, or ``None`` before the first call."""
        return self._resolved_device

    def complete(self, prompt: str, *, timeout_s: float | None = None) -> str:
        """Return the model's answer to ``prompt`` (raises ``LLMError``)."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise LLMError("prompt must be a non-empty string")
        # timeout_s is intentionally unused; see the class docstring.
        self._ensure_loaded()
        with self._lock:
            tokenizer = self._tokenizer
            if tokenizer is None:
                raise LLMError("the local model was unloaded before this call")
            templated = self._apply_chat_template(prompt, tokenizer)
            try:
                return self._generate(templated)
            except LLMError:
                raise
            except Exception as exc:
                # Device moves, OOM, unsupported dtypes and tokenizer failures
                # all degrade like a failed API call.
                raise LLMError(
                    f"local generation failed for {self.model_name!r}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

    def unload(self) -> None:
        """Drop the weights and empty the CUDA cache."""
        with self._lock:
            self._model = None
            self._tokenizer = None
            self._resolved_device = None
        gc.collect()
        try:
            import torch
        except ImportError:
            return
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                raise LLMError(
                    "The local LLM backend needs torch and transformers; install "
                    "them with `pip install transformers torch`."
                ) from exc

            device = self.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            if device.startswith("cuda"):
                if not torch.cuda.is_available():
                    raise LLMError(
                        f"device={device!r} was requested but CUDA is not available; "
                        "use device='cpu' or enable a GPU runtime."
                    )
                index = 0 if device == "cuda" else int(device.split(":", 1)[1])
                if index >= torch.cuda.device_count():
                    raise LLMError(
                        f"device={device!r} was requested but this runtime has "
                        f"{torch.cuda.device_count()} CUDA device(s)."
                    )

            # Set before loading: ``_dtype_argument`` takes the resolved device,
            # so ``dtype="auto"`` cannot fall back to the raw ``"auto"`` string
            # (which is not a CUDA device and would pick float32 on a GPU run).
            self._resolved_device = device
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name, trust_remote_code=self._trust_remote_code
                )
            except Exception as exc:
                self._resolved_device = None
                raise LLMError(
                    f"Could not load the tokenizer for {self.model_name!r}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc

            try:
                model = self._load_model(AutoModelForCausalLM, torch, device)
            except LLMError:
                self._resolved_device = None
                raise
            try:
                model = model.to(device)
                model.eval()
            except Exception as exc:
                self._resolved_device = None
                raise LLMError(
                    f"Could not move {self.model_name!r} to {device!r}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if tokenizer.pad_token_id is None:
                # Decoder-only checkpoints often ship without a pad token.
                tokenizer.pad_token = tokenizer.eos_token

            self._tokenizer = tokenizer
            self._model = model

    def _dtype_argument(self, torch, device: str):
        """``"auto"`` is device-aware, not checkpoint-driven."""
        if self.dtype == "auto":
            return torch.bfloat16 if device.startswith("cuda") else torch.float32
        return getattr(torch, self.dtype)

    def _load_model(self, model_class, torch, device: str):
        """Load the weights, tolerating the ``dtype``/``torch_dtype`` rename."""
        # ``torch_dtype`` was renamed to ``dtype``; try the new spelling first
        # and fall back for older transformers.
        dtype = self._dtype_argument(torch, device)
        try:
            return model_class.from_pretrained(
                self.model_name,
                dtype=dtype,
                trust_remote_code=self._trust_remote_code,
            )
        except TypeError as exc:
            if "dtype" not in str(exc):
                raise LLMError(
                    f"Could not load {self.model_name!r}: {type(exc).__name__}: {exc}"
                ) from exc
        try:
            return model_class.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                trust_remote_code=self._trust_remote_code,
            )
        except Exception as exc:
            raise LLMError(
                f"Could not load {self.model_name!r}: {type(exc).__name__}: {exc}"
            ) from exc

    def _apply_chat_template(self, prompt: str, tokenizer) -> str:
        if not self._use_chat_template:
            return prompt
        if not getattr(tokenizer, "chat_template", None):
            return prompt
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as exc:
            raise LLMError(f"chat template failed for {self.model_name!r}: {exc}") from exc

    def _generate(self, text: str) -> str:
        import torch

        tokenizer = self._tokenizer
        model = self._model
        device = self._resolved_device
        if tokenizer is None or model is None or device is None:
            raise LLMError("the local model is not loaded")

        # The chat template already spells out the special tokens, so they must
        # not be added a second time: a tokenizer that prepends BOS (Llama-3
        # style) would otherwise see "<|begin_of_text|>" twice.
        apply_chat_template = self._use_chat_template and getattr(
            tokenizer, "chat_template", None
        )
        encoded = tokenizer(
            text, return_tensors="pt", add_special_tokens=not apply_chat_template
        ).to(device)
        kwargs: dict = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if self.temperature > 0.0:
            kwargs["do_sample"] = True
            kwargs["temperature"] = self.temperature
        else:
            kwargs["do_sample"] = False

        with torch.inference_mode():
            output = model.generate(**encoded, **kwargs)

        prompt_length = int(encoded["input_ids"].shape[-1])
        answer = tokenizer.decode(
            output[0][prompt_length:], skip_special_tokens=True
        )
        if not answer.strip():
            raise LLMError(
                f"local model {self.model_name!r} produced an empty response"
            )
        return answer.strip()