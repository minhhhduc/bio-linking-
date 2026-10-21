"""Command-line wiring for index building, normalization and evaluation.

This module is intentionally thin: every step delegates to the library
components.  None of these commands was executed during implementation, so the
CLI is wired but not empirically validated.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from .config import FUSION_KINDS, N_SYN_VALUES, RERANK_KINDS, PipelineConfig
from .data import load_bel_jsonl
from .encoder import build_encoder
from .evaluator import evaluate_all
from .feedback import FeedbackGenerator
from .index import VectorIndex
from .kb import concepts_from_dataset, load_concepts_jsonl
from .llm import OpenAICompatibleClient
from .pipeline import EntityLinker
from .rerank import LLMReranker
from .schemas import Request


def _parse_feedback_types(value: str) -> tuple[str, ...]:
    if not value.strip():
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--encoder",
        default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        help="HuggingFace model id",
    )
    parser.add_argument(
        "--feedback-types",
        type=_parse_feedback_types,
        default=("def",),
        help="comma-separated subset of def,syn,std; empty string = baseline",
    )
    parser.add_argument("--n-syn", type=int, choices=N_SYN_VALUES, default=5)
    parser.add_argument(
        "--target-kb", default="KB", help="user-supplied KB name (std feedback)"
    )
    parser.add_argument("--top-k-retrieval", type=int, default=10)
    parser.add_argument("--top-k-concept", type=int, default=5)
    parser.add_argument("--fusion", choices=FUSION_KINDS, default="vector")
    parser.add_argument("--rerank", choices=RERANK_KINDS, default="top1")
    parser.add_argument("--llm-model", default="gpt-4o")
    parser.add_argument("--index-kb", default=None)


def _config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        encoder=args.encoder,
        feedback_types=tuple(args.feedback_types),
        n_syn=args.n_syn,
        target_kb=args.target_kb,
        top_k_retrieval=args.top_k_retrieval,
        top_k_concept=args.top_k_concept,
        fusion=args.fusion,
        rerank=args.rerank,
        llm_model=args.llm_model,
        index_kb=args.index_kb,
    )


def _encoder_info(config: PipelineConfig) -> dict[str, str]:
    return {"model": config.encoder}


def _check_index_encoder(index: VectorIndex, config: PipelineConfig) -> None:
    """Refuse a prebuilt index made with a different checkpoint."""
    stored = getattr(index, "encoder_info", None)
    if stored is None:
        print(
            "WARNING: index has no encoder metadata; pass the same "
            "--encoder used to build it.",
            file=sys.stderr,
        )
        return
    expected = _encoder_info(config)
    if stored != expected:
        raise ValueError(
            "prebuilt index encoder does not match the requested encoder: "
            f"index has {stored!r}, request has {expected!r}"
        )


def _read_records(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            records.append(record)
    return records


def _concept_names_from_index(index: VectorIndex) -> dict[str, str]:
    names: dict[str, str] = {}
    for cui, surface in zip(index.cuis, index.surfaces):
        names.setdefault(cui, surface)
    return names


def _concept_names_from_concepts(concepts) -> dict[str, str]:
    return {concept.cui: concept.name for concept in concepts}


def _build_linker(
    index: VectorIndex,
    config: PipelineConfig,
    concept_names,
    *,
    encoder=None,
):
    if encoder is None:
        encoder = build_encoder(config)

    llm = None
    if config.feedback_types or config.rerank == "llm":
        llm = OpenAICompatibleClient(config.llm_model)

    generator = None
    if config.feedback_types:
        generator = FeedbackGenerator(llm)

    reranker = None
    if config.rerank == "llm":
        reranker = LLMReranker(llm)

    return EntityLinker(
        index,
        encoder,
        config,
        feedback_generator=generator,
        reranker=reranker,
        concept_names=concept_names,
    )


def _add_kb_source(parser: argparse.ArgumentParser, *, required: bool) -> None:
    group = parser.add_mutually_exclusive_group(required=required)
    group.add_argument("--kb-jsonl", help="KB JSONL with cui/name/aliases")
    group.add_argument(
        "--data",
        help="gold BEL JSONL used to derive a surrogate KB (leakage warning)",
    )


def cmd_build_index(args: argparse.Namespace) -> int:
    config = _config(args)
    if args.kb_jsonl:
        concepts = load_concepts_jsonl(args.kb_jsonl)
    else:
        print(
            "WARNING: --data derives the KB from gold mentions; evaluating the "
            "same file leaks mentions and inflates recall. This is NOT a "
            "user-supplied KB.",
            file=sys.stderr,
        )
        concepts = concepts_from_dataset(_read_records(args.data))

    index = VectorIndex.from_concepts(
        concepts,
        build_encoder(config),
        encoder_info=_encoder_info(config),
    )
    index.save(args.out)
    print(json.dumps({"concepts": len(concepts), "entries": len(index),
                      "out": str(args.out)}, ensure_ascii=False))
    return 0


def cmd_normalize(args: argparse.Namespace) -> int:
    config = _config(args)
    index = VectorIndex.load(args.index)
    _check_index_encoder(index, config)
    linker = EntityLinker(
        index,
        build_encoder(config),
        config,
        feedback_generator=(
            FeedbackGenerator(OpenAICompatibleClient(config.llm_model))
            if config.feedback_types
            else None
        ),
        reranker=(
            LLMReranker(OpenAICompatibleClient(config.llm_model))
            if config.rerank == "llm"
            else None
        ),
        concept_names=_concept_names_from_index(index),
    )
    response = linker.link(
        Request(
            mention=args.mention,
            context=args.context,
            feedback_types=config.feedback_types,
            n_syn=config.n_syn,
        )
    )
    print(json.dumps(asdict(response), ensure_ascii=False, indent=2))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    config = _config(args)
    if args.limit is not None and args.limit < 0:
        raise ValueError("--limit must be >= 0")
    loaded = load_bel_jsonl(args.data, strict=args.strict)
    samples = list(loaded.samples)
    if args.limit is not None:
        samples = samples[: args.limit]
    if loaded.issues:
        print(
            f"NOTE: {len(loaded.issues)} data issue(s); first: "
            f"{asdict(loaded.issues[0])}",
            file=sys.stderr,
        )

    encoder = None
    if args.index:
        index = VectorIndex.load(args.index)
        _check_index_encoder(index, config)
        concept_names = _concept_names_from_index(index)
    else:
        if args.kb_jsonl:
            concepts = load_concepts_jsonl(args.kb_jsonl)
        else:
            print(
                "WARNING: --data-as-kb derives the KB from the same gold file; "
                "retrieval accuracy is leaked and not comparable to the paper.",
                file=sys.stderr,
            )
            concepts = concepts_from_dataset(_read_records(args.data))
        encoder = build_encoder(config)
        index = VectorIndex.from_concepts(
            concepts,
            encoder,
            encoder_info=_encoder_info(config),
        )
        concept_names = _concept_names_from_concepts(concepts)

    linker = _build_linker(index, config, concept_names, encoder=encoder)
    overall, by_label = evaluate_all(linker, samples)
    print(json.dumps(
        {
            "overall": asdict(overall),
            "by_label": {str(label): asdict(value) for label, value in by_label.items()},
            "data_issues": len(loaded.issues),
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m paper_method.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build-index", help="build and save the vector index")
    _add_kb_source(build, required=True)
    build.add_argument("--out", required=True, help="output directory")
    _add_common(build)
    build.set_defaults(func=cmd_build_index)

    normalize = sub.add_parser("normalize", help="link one mention/context")
    normalize.add_argument("--index", required=True, help="index directory")
    normalize.add_argument("--mention", required=True)
    normalize.add_argument("--context", required=True)
    _add_common(normalize)
    normalize.set_defaults(func=cmd_normalize)

    evaluation = sub.add_parser("evaluate", help="R@1/R@5/accuracy on gold data")
    evaluation.add_argument("--data", required=True, help="gold BEL JSONL")
    source = evaluation.add_mutually_exclusive_group(required=True)
    source.add_argument("--index", help="prebuilt index directory")
    source.add_argument("--kb-jsonl", help="KB JSONL with cui/name/aliases")
    source.add_argument(
        "--data-as-kb",
        action="store_true",
        help="derive the KB from --data (leakage warning)",
    )
    evaluation.add_argument("--limit", type=int, default=None)
    evaluation.add_argument("--strict", action="store_true")
    _add_common(evaluation)
    evaluation.set_defaults(func=cmd_evaluate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
