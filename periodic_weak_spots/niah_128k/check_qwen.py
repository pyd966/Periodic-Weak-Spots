from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from periodic_weak_spots.tokenization import leading_special_ids, load_tokenizer

from .config import ExperimentConfig
from .profiles import QWEN_MODEL_ID, verify_tokenizer_files
from .validate import validate_dataset


def check_qwen_dataset(
    dataset: str | Path,
    *,
    tokenizer_path: str | Path | None = None,
    cache_dir: str | Path | None = None,
) -> dict:
    root = Path(dataset)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    config = ExperimentConfig.from_dict(metadata["config"])
    if config.model_profile != QWEN_MODEL_ID:
        raise ValueError("check_qwen 需要使用已注册的 Qwen3.8-Flash-Next 数据集")
    hashes = verify_tokenizer_files(config.profile, local_path=tokenizer_path, cache_dir=cache_dir)
    tokenizer = load_tokenizer(
        config.tokenizer.model_id,
        config.tokenizer.revision,
        local_path=tokenizer_path,
        cache_dir=cache_dir,
    )
    leading = list(leading_special_ids(tokenizer))
    if leading:
        raise ValueError("Qwen 的完整 prompt 编码不应增加前导 special token")
    report = validate_dataset(root, tokenizer)
    cells = {}
    for phase in config.matrix.phases:
        positions = []
        values = []
        queries = set()
        lengths = set()
        with (root / "rows/key_value" / f"phase_{phase}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                positions.append(row["metadata"]["target_pair_token_start"])
                values.append(row["metadata"]["target_value_token_start"])
                queries.add(row["metadata"]["query_token_start"])
                lengths.add(row["metadata"]["prompt_token_count"])
        count = len(positions)
        total = sum(positions)
        mean = config.layout.target_pair_mean
        variance = config.phase_design.second_raw_moment - config.phase_design.mean**2
        if total != count * mean or sum((position - mean) ** 2 for position in positions) != (
            count * variance
        ):
            raise ValueError(f"phase {phase} 的目标位置均值或方差不正确")
        cells[str(phase)] = {
            "rows": count,
            "target_key_min": min(positions),
            "target_key_max": max(positions),
            "target_key_mean": total / count,
            "target_key_variance": variance,
            "target_value_mean": sum(values) / count,
            "query_token_starts": sorted(queries),
            "prompt_token_counts": sorted(lengths),
        }
    report.update(
        {
            "tokenizer_files": hashes,
            "default_model_path": config.profile.model_path,
            "leading_special_ids": leading,
            "residues": cells,
            "filler_sha256": metadata["filler_sha256"],
            "versions": {
                "python": sys.version.split()[0],
                **{
                    name: importlib.metadata.version(name)
                    for name in ("transformers", "tokenizers", "gigatoken")
                },
            },
        }
    )
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    report = check_qwen_dataset(
        args.dataset, tokenizer_path=args.tokenizer_path, cache_dir=args.cache_dir
    )
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
