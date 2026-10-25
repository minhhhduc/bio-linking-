"""Backend transformers local và prompt dịch.

Model được tải từ Hugging Face hub (hoặc từ path local nếu có) bằng
``transformers``. Không import thư viện nặng ở module level để ``--help`` và các
phần không dùng model vẫn chạy được. Không có API ngoài, không lượng tử hoá.
"""

from __future__ import annotations

import logging

from .tagging import Entity

LOGGER = logging.getLogger(__name__)

PROMPT_TEMPLATE = """You are a medical translator. Translate the Vietnamese sentence into English
while preserving entity annotations.

The sentence contains tags <e0>...</e0>, <e1>...</e1>, ... Each tag marks a
medical entity of the given type.

RULES
1. Output the English sentence with every tag kept EXACTLY ONCE, same id,
   wrapped around the English equivalent of that entity.
2. Translate according to the sentence context (natural, accurate medical English).
3. Each tagged phrase must be one contiguous phrase that translates that
   entity; inflection or word order may change to fit the sentence.
4. Tag ONLY the term itself, no articles/prepositions/neighbors
   ("a <e1>virus</e1>", not "<e1>a virus</e1>").
5. Do NOT merge two entities into one word. If natural English would merge them
   (e.g. "viral meningitis"), rewrite so each concept stays separate
   ("meningitis caused by a virus").
6. Do not add, drop, reorder or nest tags. No other markup.
7. Do not translate the codes or labels.

ENTITIES
{entities}

EXAMPLE
Vietnamese: <e0>viêm màng não</e0> do <e1>virus</e1> : biến chứng hiếm gặp , xảy ra khi <e2>virus</e2> xâm nhập vào hệ thống thần kinh trung ương của cơ thể ( não và tủy sống ) .
English: <e0>Meningitis</e0> caused by a <e1>virus</e1>: a rare complication that occurs when the <e2>virus</e2> enters the central nervous system of the body (brain and spinal cord).

NOW TRANSLATE
Vietnamese: {vi_tagged}
{retry_feedback}

Return JSON only: {{"en_tagged": "<tagged English sentence>"}}"""


class BackendError(RuntimeError):
    """Backend không khởi tạo hoặc sinh lỗi không phải OOM."""


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
) -> str:
    if not entities:
        return "(no entities in this sentence)"
    return "\n".join(
        f"- e{index}: {entity.vi_term!r} | label={entity.label}"
        for index, entity in enumerate(entities)
    )


def build_prompt(
    entities: list[Entity] | tuple[Entity, ...],
    vi_tagged: str,
    retry_feedback: str = "",
) -> str:
    return PROMPT_TEMPLATE.format(
        entities=build_entities_block(entities),
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
    """transformers + generate, batch thật, full precision."""

    DTYPES = ("float16", "bfloat16", "float32")

    def __init__(
        self,
        model: str,
        *,
        max_new_tokens: int = 2048,
        max_seq_len: int | None = None,
        dtype: str = "bfloat16",
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise BackendError("model must be a Hugging Face repo id or local path")
        if dtype not in self.DTYPES:
            raise BackendError(
                f"unsupported dtype {dtype!r}; choose one of {self.DTYPES}"
            )
        if max_new_tokens < 1:
            raise BackendError("max_new_tokens must be >= 1")
        if max_seq_len is not None and max_seq_len < 1:
            raise BackendError("max_seq_len must be >= 1 or None")
        try:
            import torch
            import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise BackendError("requires torch and transformers") from exc

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.max_seq_len = max_seq_len
        self.cuda = torch.cuda.is_available()
        if not self.cuda:
            LOGGER.warning("torch has no CUDA; a 30B model will not fit on CPU")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            trust_remote_code=False,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        torch_dtype = getattr(torch, dtype)
        kwargs = {
            "trust_remote_code": False,
            # A100 80GB: để accelerate đặt thẳng lên GPU, tránh đỉnh RAM CPU.
            "device_map": "auto" if self.cuda else None,
        }
        # transformers >= 4.56 dùng `dtype`; bản cũ dùng `torch_dtype`.
        major, minor = (int(part) for part in transformers.__version__.split(".")[:2])
        kwargs["dtype" if (major, minor) >= (4, 56) else "torch_dtype"] = torch_dtype
        self.model = AutoModelForCausalLM.from_pretrained(model, **kwargs)
        self.model.eval()
        self.device = self.model.device

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
        limit = self.max_seq_len
        if limit is None:
            limit = getattr(self.model.config, "max_position_embeddings", None)
        if limit and input_length + self.max_new_tokens > limit:
            raise BackendError(
                f"prompt is {input_length} tokens; +{self.max_new_tokens} generation "
                f"exceeds max_seq_len={limit}"
            )
        with self.torch.inference_mode():
            outputs = self.model.generate(
                **encoded,
                do_sample=False,  # temperature = 0
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = outputs[:, input_length:]
        return self.tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
