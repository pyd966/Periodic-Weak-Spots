from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .profiles import verify_tokenizer_files
from .score_vllm import generate, load_rows


ORACLE_CLASS_OVERRIDES = {
    "Qwen4ExpForConditionalGeneration": (
        "periodic_weak_spots.niah_128k.oracle_vllm:"
        "OracleQwen4ExpForConditionalGeneration"
    ),
    "Qwen4ExpForCausalLM": (
        "periodic_weak_spots.niah_128k.oracle_vllm:OracleQwen4ExpForCausalLM"
    ),
}


def run_oracle(
    dataset: str | Path,
    model: str | None,
    *,
    tensor_parallel_size: int,
    max_model_len: int,
    enable_expert_parallel: bool,
    max_tokens: int,
    batch_size: int,
    progress: str | Path | None,
    results: str | Path,
) -> dict[str, Any]:
    dataset_path = Path(dataset)
    metadata_bytes = (dataset_path / "metadata.json").read_bytes()
    metadata = json.loads(metadata_bytes)
    config = ExperimentConfig.from_dict(metadata["config"])
    model_path = model or config.profile.model_path
    tokenizer_files = {}
    if config.profile.use_chat_template:
        tokenizer_files = verify_tokenizer_files(config.profile, local_path=model_path)
    model_config_sha256 = None
    model_config_path = Path(model_path) / "config.json"
    if model_config_path.is_file():
        model_config_sha256 = hashlib.sha256(model_config_path.read_bytes()).hexdigest()
    run_metadata = {
        "dataset_path": str(dataset_path.resolve()),
        "dataset_metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
        "dataset_metadata": metadata,
        "model_profile": config.model_profile,
        "tokenizer_id": config.tokenizer.model_id,
        "tokenizer_revision": config.tokenizer.revision,
        "tokenizer_files": tokenizer_files,
        "model_config_sha256": model_config_sha256,
        "oracle_indexer": {
            "class_overrides": ORACLE_CLASS_OVERRIDES,
            "scope": "final_query_rows",
            "score": "sum_relu_qk_scores_over_indexer_heads_then_four_token_blocks",
            "key_representation": "QSA_indexer_k_layernorm_and_RoPE_output",
            "query_representation": "QSA_indexer_q_layernorm_and_RoPE_output",
            "historical_key_store": "CPU_bfloat16",
        },
    }
    rows = load_rows(dataset_path)
    result = generate(
        model_path,
        rows,
        max_tokens=max_tokens,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        enable_expert_parallel=enable_expert_parallel,
        batch_size=batch_size,
        progress_path=progress,
        progress_run_metadata=run_metadata,
        enforce_eager=True,
        enable_prefix_caching=False,
        model_class_overrides=ORACLE_CLASS_OVERRIDES,
    )
    result["schema_version"] = 2
    result["run"].update(run_metadata)
    results_path = Path(results)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m periodic_weak_spots.niah_128k.oracle_score"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=131072)
    parser.add_argument("--enable-expert-parallel", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    result = run_oracle(
        args.dataset,
        args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        enable_expert_parallel=args.enable_expert_parallel,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        progress=args.progress,
        results=args.results,
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
