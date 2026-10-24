"""Backend model local (vLLM / transformers / llama-cpp) và prompt dịch.

Không import thư viện nặng ở module level để ``--help`` và các phần không dùng
model vẫn chạy được. Không có API ngoài, không tự tải model.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .tagging import Entity

LOGGER = logging.getLogger(__name__)

PROMPT_TEMPLATE = """You are a medical translator. Translate the Vietnamese sentence into English
while preserving entity annotations.

The sentence contains tags <e0>...</e0>, <e1>...</e1>, ... Each tag marks a
medical entity linked to a UMLS concept.

RULES
1. Output the English sentence with every tag kept EXACTLY ONCE, same id,
   wrapped around the English equivalent of that entity.
2. Translate according to the sentence context (natural, accurate medical English).
3. Each tagged phrase must be one contiguous phrase and must denote the given
   UMLS concept. Prefer one of the listed English names; inflection or word
   order may change if the context requires.
4. Tag ONLY the term itself, no articles/prepositions/neighbors
   ("a <e1>virus</e1>", not "<e1>a virus</e1>").
5. Do NOT merge two entities into one word. If natural English would merge them
   (e.g. "viral meningitis"), rewrite so each concept stays separate
   ("meningitis caused by a virus").
6. Do not add, drop, reorder or nest tags. No other markup.
7. Do not translate CUI, ICD codes, or labels.

ENTITIES
{entities}

EXAMPLE
Vietnamese: <e0>viêm màng não</e0> do <e1>virus</e1> : biến chứng hiếm gặp , xảy ra khi <e2>virus</e2> xâm nhập vào hệ thống thần kinh trung ương của cơ thể ( não và tủy sống ) .
English: <e0>Meningitis</e0> caused by a <e1>virus</e1>: a rare complication that occurs when the <e2>virus</e2> enters the central nervous system of the body (brain and spinal cord).

NOW TRANSLATE
Vietnamese: {vi_tagged}
{retry_feedback}

Return JSON only: {{"en_tagged": "<tagged English sentence>"}}"""

JSON_SCHEMA = {
    "type": "object",
    "properties": {"en_tagged": {"type": "string"}},
    "required": ["en_tagged"],
    "additionalProperties": False,
}


class BackendError(RuntimeError):
    """Backend không khởi tạo hoặc sinh lỗi không phải OOM."""


class OOMError(RuntimeError):
    """Batch quá lớn; pipeline sẽ giảm batch và chạy lại."""


def is_oom(exc: BaseException) -> bool:
    if isinstance(exc, MemoryError):
        return True
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return (
        "outofmemory" in name
        or "out of memory" in text
        or "cuda oom" in text
        or "cublas_status_alloc_failed" in text
        or "can't allocate memory" in text
        or "failed to allocate" in text
        or "bad_alloc" in text
        or "alloc_cpu" in text
        or "no available memory" in text
        or "kv cache" in text
    )


def format_retry_feedback(reasons: list[str] | tuple[str, ...]) -> str:
    if not reasons:
        return ""
    lines = "\n".join(f"- {reason}" for reason in reasons)
    return f"RETRY FEEDBACK - fix these problems:\n{lines}"


def build_entities_block(
    entities: list[Entity] | tuple[Entity, ...],
    hints: dict[str, tuple[str, ...]],
) -> str:
    if not entities:
        return "(no entities in this sentence)"
    lines: list[str] = []
    for index, entity in enumerate(entities):
        names = hints.get(entity.cui, ())
        english = "; ".join(names) if names else "(no English name found)"
        icd10 = ", ".join(entity.icd10) if entity.icd10 else "-"
        lines.append(
            f"- e{index}: {entity.vi_term!r} | CUI={entity.cui} | "
            f"ICD10={icd10} | label={entity.label} | EN names: {english}"
        )
    return "\n".join(lines)


def build_prompt(
    entities: list[Entity] | tuple[Entity, ...],
    vi_tagged: str,
    hints: dict[str, tuple[str, ...]],
    retry_feedback: str = "",
) -> str:
    return PROMPT_TEMPLATE.format(
        entities=build_entities_block(entities, hints),
        vi_tagged=vi_tagged,
        retry_feedback=retry_feedback,
    )


def _apply_chat_template(tokenizer, prompt: str) -> str:
    template = getattr(tokenizer, "chat_template", None)
    if not template:
        raise BackendError("tokenizer has no chat_template")
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        # Qwen3 template phải tắt được thinking. Nếu template có biến này mà
        # tokenizer vẫn từ chối, không fallback im lặng.
        if "enable_thinking" in template:
            raise BackendError(
                "chat template supports enable_thinking but tokenizer rejected it"
            )
        LOGGER.warning(
            "chat template has no enable_thinking flag; thinking may stay on"
        )
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


class HFBackend:
    """transformers + generate, batch thật."""

    def __init__(
        self,
        model: str,
        *,
        gpu_mem_util: float = 0.90,
        tp: int = 1,
    ) -> None:
        if not Path(model).exists():
            raise BackendError(f"model path not found: {model}")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise BackendError(
                "hf backend requires torch and transformers"
            ) from exc

        self.torch = torch
        self.cuda = torch.cuda.is_available()
        self.device = "cuda" if self.cuda else "cpu"
        if self.cuda and tp != 1:
            LOGGER.warning("HF backend does not implement tensor parallelism; --tp ignored")
        if gpu_mem_util != 0.90:
            LOGGER.warning("HF backend does not enforce --gpu-mem-util")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            local_files_only=True,
            trust_remote_code=False,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        dtype = torch.float16 if self.cuda else torch.bfloat16
        if not self.cuda:
            LOGGER.warning(
                "torch is CPU-only; Qwen 27B needs a GPU in practice"
            )
        kwargs = {
            "local_files_only": True,
            "trust_remote_code": False,
        }
        try:
            # transformers >= 4.56 dùng `dtype`; bản cũ dùng `torch_dtype`.
            self.model = AutoModelForCausalLM.from_pretrained(
                model, dtype=dtype, **kwargs
            )
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                model, torch_dtype=dtype, **kwargs
            )
        self.model.to(self.device)
        self.model.eval()

    def generate_batch(self, prompts: list[str]) -> list[str]:
        texts = [_apply_chat_template(self.tokenizer, prompt) for prompt in prompts]
        encoded = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        input_length = encoded["input_ids"].shape[1]
        max_positions = getattr(self.model.config, "max_position_embeddings", None)
        if max_positions and input_length + 512 > max_positions:
            raise BackendError(
                f"prompt is {input_length} tokens; +512 generation exceeds "
                f"model max_position_embeddings={max_positions}"
            )
        with self.torch.inference_mode():
            outputs = self.model.generate(
                **encoded,
                do_sample=False,  # temperature = 0
                max_new_tokens=512,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = outputs[:, input_length:]
        return self.tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )


class VLLMBackend:
    """vLLM offline ``LLM`` + guided JSON nếu phiên bản hỗ trợ."""

    def __init__(
        self,
        model: str,
        *,
        gpu_mem_util: float = 0.90,
        tp: int = 1,
    ) -> None:
        if not Path(model).exists():
            raise BackendError(f"model path not found: {model}")
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise BackendError("vllm backend requires vllm") from exc

        self.SamplingParams = SamplingParams
        self.llm = LLM(
            model=model,
            tensor_parallel_size=tp,
            gpu_memory_utilization=gpu_mem_util,
            max_model_len=8192,
            dtype="bfloat16",
            trust_remote_code=False,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling = self._make_sampling()

    def _make_sampling(self):
        guided = None
        try:
            from vllm.sampling_params import GuidedDecodingParams

            try:
                guided = GuidedDecodingParams(json=JSON_SCHEMA)
            except TypeError:
                guided = GuidedDecodingParams(json=json.dumps(JSON_SCHEMA))
        except (ImportError, TypeError, ValueError) as exc:
            LOGGER.warning("vLLM GuidedDecodingParams unavailable: %s", exc)
            guided = None

        if guided is not None:
            try:
                return self.SamplingParams(
                    temperature=0.0,
                    max_tokens=512,
                    guided_decoding=guided,
                )
            except TypeError:
                pass
        try:
            return self.SamplingParams(
                temperature=0.0,
                max_tokens=512,
                guided_json=JSON_SCHEMA,
            )
        except TypeError:
            LOGGER.warning("vLLM guided JSON unavailable; falling back to regex parse")
            return self.SamplingParams(temperature=0.0, max_tokens=512)

    def generate_batch(self, prompts: list[str]) -> list[str]:
        texts = [_apply_chat_template(self.tokenizer, prompt) for prompt in prompts]
        requests: list[object] = texts
        try:
            from vllm.inputs import TokensPrompt

            requests = [
                TokensPrompt(
                    prompt_token_ids=self.tokenizer(
                        text, add_special_tokens=False
                    )["input_ids"]
                )
                for text in texts
            ]
        except (ImportError, AttributeError, TypeError):
            requests = texts
        outputs = self.llm.generate(requests, self.sampling)
        return [output.outputs[0].text for output in outputs]


class LlamaCppBackend:
    """llama-cpp-python cho model GGUF."""

    def __init__(
        self,
        model: str,
        *,
        gpu_mem_util: float = 0.90,
        tp: int = 1,
    ) -> None:
        if not Path(model).exists():
            raise BackendError(f"model path not found: {model}")
        if tp != 1:
            LOGGER.warning("llama.cpp backend does not implement --tp; ignored")
        if gpu_mem_util != 0.90:
            LOGGER.warning("llama.cpp backend does not enforce --gpu-mem-util")
        try:
            from llama_cpp import Llama, LlamaGrammar
        except ImportError as exc:
            raise BackendError("llamacpp backend requires llama-cpp-python") from exc

        self.grammar = None
        try:
            self.grammar = LlamaGrammar.from_json_schema(json.dumps(JSON_SCHEMA))
        except (AttributeError, TypeError, ValueError):
            LOGGER.warning("llama.cpp JSON grammar unavailable; using response_format")

        try:
            import torch

            gpu_layers = -1 if torch.cuda.is_available() else 0
        except ImportError:
            gpu_layers = -1
        self.llm = Llama(
            model_path=model,
            n_ctx=8192,
            n_gpu_layers=gpu_layers,
            verbose=False,
        )

    def generate_batch(self, prompts: list[str]) -> list[str]:
        results: list[str] = []
        for prompt in prompts:
            kwargs: dict[str, object] = {
                "temperature": 0.0,
                "max_tokens": 512,
            }
            if self.grammar is not None:
                kwargs["grammar"] = self.grammar
            else:
                kwargs["response_format"] = {"type": "json_object"}

            messages = [{"role": "user", "content": prompt}]
            try:
                response = self.llm.create_chat_completion(
                    messages=messages,
                    chat_template_kwargs={"enable_thinking": False},
                    **kwargs,
                )
            except TypeError:
                LOGGER.warning(
                    "llama.cpp chat_template_kwargs unsupported; thinking may stay on"
                )
                response = self.llm.create_chat_completion(messages=messages, **kwargs)
            message = response["choices"][0]["message"]
            content = message.get("content") or message.get("reasoning_content") or ""
            results.append(content)
        return results


def make_backend(
    name: str,
    model: str,
    *,
    gpu_mem_util: float = 0.90,
    tp: int = 1,
):
    backend = name.lower()
    if backend == "hf":
        return HFBackend(model, gpu_mem_util=gpu_mem_util, tp=tp)
    if backend == "vllm":
        return VLLMBackend(model, gpu_mem_util=gpu_mem_util, tp=tp)
    if backend == "llamacpp":
        return LlamaCppBackend(model, gpu_mem_util=gpu_mem_util, tp=tp)
    raise BackendError(f"unknown backend: {name}")
