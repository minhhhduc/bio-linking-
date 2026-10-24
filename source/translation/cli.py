"""CLI cho pipeline dịch local."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .pipeline import PipelineConfig, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Dịch NER y khoa VI -> EN bằng model Qwen local."
    )
    parser.add_argument("--model", required=True, help="đường dẫn model Qwen local")
    parser.add_argument(
        "--backend",
        choices=("vllm", "hf", "llamacpp"),
        default="hf",
        help="backend local (mặc định: hf)",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gpu-mem-util", type=float, default=0.90)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--umls", required=True, help="đường dẫn MRCONSO.RRF")
    parser.add_argument("--sap-thr", type=float, default=0.85)
    parser.add_argument("--in", dest="input_path", required=True, help="JSONL nguồn")
    parser.add_argument("--out", dest="output_path", required=True, help="JSONL output")
    parser.add_argument(
        "--review",
        default=None,
        help="JSONL record fail (mặc định: review.jsonl cạnh --out)",
    )
    parser.add_argument(
        "--umls-cache",
        default=None,
        help="pickle cache hint (mặc định: umls_hints.pkl cạnh --out)",
    )
    parser.add_argument(
        "--sap-model",
        default=os.environ.get("SAPBERT_MODEL"),
        help="đường dẫn SapBERT local; mặc định lấy SAPBERT_MODEL",
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
    if args.max_retries < 0:
        parser.error("--max-retries must be >= 0")
    if not 0.0 < args.gpu_mem_util <= 1.0:
        parser.error("--gpu-mem-util must be in (0, 1]")
    if not 0.0 < args.sap_thr <= 1.0:
        parser.error("--sap-thr must be in (0, 1]")
    if args.tp < 1:
        parser.error("--tp must be >= 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")

    model_path = Path(args.model)
    if not model_path.exists():
        parser.error(f"model path not found: {model_path}")
    input_path = Path(args.input_path)
    if not input_path.exists():
        parser.error(f"input file not found: {input_path}")
    umls_path = Path(args.umls)
    if not umls_path.exists():
        parser.error(f"UMLS MRCONSO not found: {umls_path}")

    output_path = Path(args.output_path)
    review_path = (
        Path(args.review)
        if args.review
        else output_path.parent / "review.jsonl"
    )
    umls_cache = (
        Path(args.umls_cache)
        if args.umls_cache
        else output_path.parent / "umls_hints.pkl"
    )

    if output_path.resolve() == input_path.resolve():
        parser.error("--out must differ from --in")
    if review_path.resolve() == output_path.resolve():
        parser.error("--review must differ from --out")
    if review_path.resolve() == input_path.resolve():
        parser.error("--review must differ from --in")
    if output_path.resolve() == umls_path.resolve():
        parser.error("--out must differ from --umls")
    if review_path.resolve() == umls_path.resolve():
        parser.error("--review must differ from --umls")
    if umls_cache.resolve() in {
        input_path.resolve(),
        output_path.resolve(),
        review_path.resolve(),
        umls_path.resolve(),
    }:
        parser.error("--umls-cache must differ from --in/--out/--review/--umls")

    config = PipelineConfig(
        model=str(model_path),
        backend=args.backend,
        batch_size=args.batch_size,
        gpu_mem_util=args.gpu_mem_util,
        tp=args.tp,
        umls_path=str(umls_path),
        sap_thr=args.sap_thr,
        input_path=str(input_path),
        output_path=str(output_path),
        review_path=str(review_path),
        umls_cache=str(umls_cache),
        sap_model=args.sap_model,
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
