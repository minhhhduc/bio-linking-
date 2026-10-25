"""CLI cho pipeline dịch local."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import PipelineConfig, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dịch NER y khoa VI -> EN bằng model Qwen local."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Hugging Face repo id (vd Qwen/Qwen3-32B) hoặc path local",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="prompt + generation cap; mặc định lấy max_position_embeddings của model",
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
        help="kiểu weight khi load bằng transformers",
    )
    parser.add_argument("--in", dest="input_path", required=True, help="JSONL nguồn")
    parser.add_argument("--out", dest="output_path", required=True, help="JSONL output")
    parser.add_argument(
        "--review",
        default=None,
        help="JSONL record fail (mặc định: review.jsonl cạnh --out)",
    )
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be >= 1")
    if args.max_seq_len is not None and args.max_seq_len < 1:
        parser.error("--max-seq-len must be >= 1")
    if args.max_retries < 0:
        parser.error("--max-retries must be >= 0")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")

    input_path = Path(args.input_path)
    if not input_path.exists():
        parser.error(f"input file not found: {input_path}")

    output_path = Path(args.output_path)
    review_path = (
        Path(args.review)
        if args.review
        else output_path.parent / "review.jsonl"
    )

    if output_path.resolve() == input_path.resolve():
        parser.error("--out must differ from --in")
    if review_path.resolve() == output_path.resolve():
        parser.error("--review must differ from --out")
    if review_path.resolve() == input_path.resolve():
        parser.error("--review must differ from --in")

    config = PipelineConfig(
        model=args.model,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        max_seq_len=args.max_seq_len,
        dtype=args.dtype,
        input_path=str(input_path),
        output_path=str(output_path),
        review_path=str(review_path),
        max_retries=args.max_retries,
        resume=not args.no_resume,
        limit=args.limit,
    )

    try:
        stats = run_pipeline(config)
    except Exception as exc:  # CLI trả lỗi rõ, không nuốt traceback quá mức.
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(stats.as_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
