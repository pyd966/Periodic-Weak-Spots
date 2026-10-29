"""``python -m periodic_weak_spots.niah_128k {build,validate,score}``."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# The NIAH haystack of Paul Graham essays used for the paper (not redistributed).
PAPER_FILLER_SHA256 = "51eaf041f492d8fdfdcf87be8756d23888eda0d3cab6816b3c3f11e7682de337"


def _print(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m periodic_weak_spots.niah_128k")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="generate a balanced 128K NIAH dataset")
    build.add_argument("--config", type=Path, required=True, help="configs/niah_128k/<role>.json")
    build.add_argument(
        "--filler-file",
        type=Path,
        required=True,
        help="filler text for padding, alignment, and tail",
    )
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--cache-dir")
    build.add_argument("--source-file", type=Path, help="本地固定版本的 train.txt.gz")
    build.add_argument("--tokenizer-path", type=Path, help="local copy of the pinned tokenizer")
    build.add_argument("--overwrite", action="store_true")
    validate = commands.add_parser("validate", help="independently retokenize a built dataset")
    validate.add_argument("--dataset", type=Path, required=True, help="directory written by build")
    validate.add_argument("--cache-dir")
    validate.add_argument("--tokenizer-path", type=Path)
    validate.add_argument("--report", type=Path, help="保存检查结果")
    score = commands.add_parser("score", help="greedy vLLM decoding + answer-prefix accuracy")
    score.add_argument("--dataset", type=Path, required=True, help="directory written by build")
    score.add_argument("--model", help="本地模型目录；默认使用注册的模型目录")
    score.add_argument("--tensor-parallel-size", type=int, default=1)
    score.add_argument("--max-model-len", type=int, default=131072)
    score.add_argument("--enable-expert-parallel", action="store_true")
    score.add_argument("--max-tokens", type=int, default=8)
    score.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="每次提交给 vLLM 的 prompt 数量",
    )
    score.add_argument(
        "--progress",
        type=Path,
        help="追加写入评分进度 JSONL，供 monitor 页面读取",
    )
    score.add_argument("--results", type=Path, required=True, help="保存逐条结果与统计的 JSON 文件")
    args = parser.parse_args(argv)

    # Deferred imports keep each subcommand to its own dependencies (vLLM only for score).
    from periodic_weak_spots.tokenization import load_tokenizer

    from .config import ExperimentConfig, load_config
    from .profiles import verify_tokenizer_files

    if args.command == "score":
        from .score_vllm import generate, load_rows

        metadata_bytes = (args.dataset / "metadata.json").read_bytes()
        metadata = json.loads(metadata_bytes)
        config = ExperimentConfig.from_dict(metadata["config"])
        model_path = args.model or config.profile.model_path
        tokenizer_files = {}
        if config.profile.use_chat_template:
            tokenizer_files = verify_tokenizer_files(config.profile, local_path=model_path)
        model_config_sha256 = None
        model_config_path = Path(model_path) / "config.json"
        if model_config_path.is_file():
            model_config_sha256 = hashlib.sha256(model_config_path.read_bytes()).hexdigest()
        run_metadata = {
            "dataset_path": str(args.dataset.resolve()),
            "dataset_metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
            "dataset_metadata": metadata,
            "model_profile": config.model_profile,
            "tokenizer_id": config.tokenizer.model_id,
            "tokenizer_revision": config.tokenizer.revision,
            "tokenizer_files": tokenizer_files,
            "model_config_sha256": model_config_sha256,
        }
        args.results.parent.mkdir(parents=True, exist_ok=True)
        rows = load_rows(args.dataset)
        result = generate(
            model_path,
            rows,
            max_tokens=args.max_tokens,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            enable_expert_parallel=args.enable_expert_parallel,
            batch_size=args.batch_size,
            progress_path=args.progress,
            progress_run_metadata=run_metadata,
        )
        result["schema_version"] = 2
        result["run"].update(run_metadata)
        args.results.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
        _print(result["summary"])
        return

    if args.command == "validate":
        from .validate import validate_dataset

        metadata = json.loads((args.dataset / "metadata.json").read_text(encoding="utf-8"))
        config = ExperimentConfig.from_dict(metadata["config"])
        spec = config.tokenizer
        verify_tokenizer_files(
            config.profile, local_path=args.tokenizer_path, cache_dir=args.cache_dir
        )
        tokenizer = load_tokenizer(
            spec.model_id, spec.revision, local_path=args.tokenizer_path, cache_dir=args.cache_dir
        )
        report = validate_dataset(args.dataset, tokenizer)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
        _print(report)
        return

    from .bindings import collect_context_stable_token_candidates, generate_bindings
    from .build import publish_dataset
    from .schedule import sentence_capacity_requirements
    from .source import curate_sentence_bins, load_pinned_source

    config = load_config(args.config)
    verify_tokenizer_files(config.profile, local_path=args.tokenizer_path, cache_dir=args.cache_dir)
    tokenizer = load_tokenizer(
        config.tokenizer.model_id,
        config.tokenizer.revision,
        local_path=args.tokenizer_path,
        cache_dir=args.cache_dir,
    )
    candidates = collect_context_stable_token_candidates(
        tokenizer,
        min_token_id=config.candidate_pool.min_token_id,
        max_characters=config.candidate_pool.max_characters,
    )
    # One binding per row index; every phase uses the full set, so residues share their records.
    bindings = generate_bindings(
        candidates,
        count=config.phase_design.rows_per_phase,
        pair_count=config.layout.pair_count,
        target_index=config.layout.target_index,
        seed=config.binding_seed,
    )
    bins, source_report = curate_sentence_bins(
        load_pinned_source(
            config.source,
            cache_dir=args.cache_dir,
            local_file=None if args.source_file is None else str(args.source_file),
        ),
        tokenizer,
        sentence_capacity_requirements(config.phase_design.rows_per_phase),
        spec=config.source,
    )
    filler = args.filler_file.read_bytes()
    metadata = publish_dataset(
        args.output,
        tokenizer,
        bins,
        bindings,
        filler=filler.decode("utf-8"),
        source_report=source_report,
        config=config,
        overwrite=args.overwrite,
    )
    # Any filler builds a valid dataset; the hash only reports fidelity to the paper's prompts.
    filler_sha256 = hashlib.sha256(filler).hexdigest()
    _print(
        {
            "dataset_id": metadata["dataset_id"],
            "total_rows": metadata["matrix"]["total_rows"],
            "filler_sha256": filler_sha256,
            "filler_matches_paper": filler_sha256 == PAPER_FILLER_SHA256,
        }
    )


if __name__ == "__main__":
    main()
