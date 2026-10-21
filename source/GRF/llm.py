"""OpenAI-compatible chat completion client using only the standard library.

The pipeline deliberately has no retry logic: a failed feedback or re-rank
call degrades to the baseline path (see ``docs/architecture.md`` §6).
"""

from __future__ import annotations

import http.client
import json
import math
import os
import urllib.error
import urllib.request
from typing import Protocol

DEFAULT_BASE_URL = "https://api.openai.com/v1"
_MAX_ERROR_BODY_CHARS = 500


class LLMError(RuntimeError):
    """Raised when an LLM request cannot be built, sent, or parsed."""


class LLMClient(Protocol):
    def complete(self, prompt: str, *, timeout_s: float | None = None) -> str:
        """Return the model's text response for ``prompt``."""
        ...


def _positive_float(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise LLMError(f"{name} must be a number, got {value!r}")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise LLMError(f"{name} must be a number, got {value!r}") from exc
    if not math.isfinite(number) or number <= 0:
        raise LLMError(f"{name} must be a positive finite number, got {value!r}")
    return number


def _error_body(exc: urllib.error.HTTPError) -> str:
    """Best-effort body snippet; never includes request headers or the key."""
    try:
        raw = exc.read() or b""
    except Exception:  # pragma: no cover - defensive, body may be unreadable
        return "<unreadable body>"
    text = raw.decode("utf-8", "replace").strip()
    if not text:
        return "<empty body>"
    if len(text) > _MAX_ERROR_BODY_CHARS:
        return text[:_MAX_ERROR_BODY_CHARS] + "..."
    return text


class OpenAICompatibleClient:
    """POST ``{"model", "messages", "temperature"}`` to ``/chat/completions``.

    ``api_key`` and ``base_url`` fall back to ``OPENAI_API_KEY`` and
    ``OPENAI_BASE_URL``; the base URL defaults to the public OpenAI endpoint.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        timeout_s: float = 30.0,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise LLMError("model must be a non-empty string, e.g. 'gpt-4o'")
        self.model = model.strip()

        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY", "")
        if not isinstance(api_key, str):
            raise LLMError("api_key must be a string or None")
        self._api_key = api_key.strip() or None

        if base_url is None:
            base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or DEFAULT_BASE_URL
        elif not isinstance(base_url, str) or not base_url.strip():
            raise LLMError("base_url must be a non-empty URL, e.g. 'https://api.openai.com/v1'")
        self.base_url = base_url.strip().rstrip("/")

        if isinstance(temperature, bool):
            raise LLMError(f"temperature must be a number, got {temperature!r}")
        try:
            temperature_value = float(temperature)
        except (TypeError, ValueError) as exc:
            raise LLMError(f"temperature must be a number, got {temperature!r}") from exc
        if not math.isfinite(temperature_value):
            raise LLMError(f"temperature must be finite, got {temperature!r}")
        self.temperature = temperature_value

        self.timeout_s = _positive_float(timeout_s, "timeout_s")

    def complete(self, prompt: str, *, timeout_s: float | None = None) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise LLMError("prompt must be a non-empty string")

        timeout = self.timeout_s if timeout_s is None else _positive_float(timeout_s, "timeout_s")

        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        payload = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        url = self.base_url.rstrip("/") + "/chat/completions"
        request = urllib.request.Request(url, data=payload, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            raise LLMError(
                f"LLM request failed with HTTP {exc.code} {exc.reason}: {_error_body(exc)}"
            ) from exc
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            raise LLMError(
                f"LLM request failed (network error or timeout): {reason}. "
                f"Check connectivity and base_url={self.base_url!r}."
            ) from exc
        except TimeoutError as exc:
            raise LLMError(
                f"LLM request timed out after {timeout:g}s (base_url={self.base_url!r})."
            ) from exc
        except (http.client.HTTPException, OSError) as exc:
            raise LLMError(
                "LLM response could not be read "
                f"({type(exc).__name__}: {exc}). Check connectivity and "
                f"base_url={self.base_url!r}."
            ) from exc

        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LLMError(f"LLM response was not valid UTF-8 JSON: {exc}") from exc

        choices = data.get("choices") if isinstance(data, dict) else None
        if not isinstance(choices, list) or not choices:
            raise LLMError("LLM response is missing a non-empty 'choices' list")

        first = choices[0]
        message = first.get("message") if isinstance(first, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise LLMError("LLM response has empty choices[0].message.content")

        return content
