from __future__ import annotations

import importlib.metadata
import json
import math
import time
from collections.abc import Sequence
from contextlib import nullcontext
from itertools import pairwise
from pathlib import Path
from typing import Any

from periodic_weak_spots.tokenization import token_ids_sha256

from .layout import PAIR_FORM, render_query
from .reverse_query import QUERY_DIRECTION as REVERSE_DIRECTION
from .reverse_query import render_reverse_query

TOP_LOGPROBS = 10
FORWARD_DIRECTION = "key_to_value"


def expected_query_and_answer(row: dict[str, Any], key: str, value: str) -> tuple[str, str]:
    """按 row 的 query 方向给出 query 文本与 gold：正向问 key 答 value，反向问 value 答 key。"""

    direction = row.get("query_direction", FORWARD_DIRECTION)
    if direction == FORWARD_DIRECTION:
        query, answer = render_query(key), value
    elif direction == REVERSE_DIRECTION:
        query, answer = render_reverse_query(value), key
    else:
        raise ValueError(f"未知的 query 方向：{direction}")
    if "chat_prefix" in row["components"]:
        query = query.rstrip()
    return query, answer


def load_rows(output_root: str | Path) -> list[dict[str, Any]]:
    """All prompt rows of a built dataset, in phase order."""

    root = Path(output_root)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for phase in metadata["matrix"]["phases"]:
        path = root / "rows" / PAIR_FORM / f"phase_{phase}.jsonl"
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle)
    return rows


def is_correct(completion: str, gold: str) -> bool:
    # 使用论文的答案前缀准确率。
    return completion.strip().startswith(gold)


def wilson_interval(correct: int, total: int, *, z: float = 1.96) -> tuple[float, float]:
    if total <= 0 or correct < 0 or correct > total:
        raise ValueError("Wilson 区间的计数不正确")
    proportion = correct / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return center - radius, center + radius


def summarize(rows: Sequence[dict[str, Any]], completions: Sequence[str]) -> dict[str, Any]:
    """Accuracy per residue mod 8 (a row's ``phase`` field) and the best-minus-worst gap."""

    by_residue: dict[int, list[bool]] = {}
    for row, completion in zip(rows, completions, strict=True):
        by_residue.setdefault(row["phase"], []).append(is_correct(completion, row["answer"][0]))
    accuracy = {residue: sum(hits) / len(hits) for residue, hits in sorted(by_residue.items())}
    counts = {residue: sum(hits) for residue, hits in sorted(by_residue.items())}
    sizes = {residue: len(hits) for residue, hits in sorted(by_residue.items())}
    wilson95 = {
        residue: wilson_interval(counts[residue], sizes[residue]) for residue in sorted(by_residue)
    }
    pair_differences = {
        residue: abs(accuracy[residue] - accuracy[residue + 4]) for residue in range(4)
    }
    total = [hit for hits in by_residue.values() for hit in hits]
    return {
        "rows": len(total),
        "answer_prefix_accuracy": sum(total) / len(total),
        "answer_prefix_accuracy_by_residue": {str(r): value for r, value in accuracy.items()},
        "correct_count_by_residue": {str(r): value for r, value in counts.items()},
        "sample_count_by_residue": {str(r): value for r, value in sizes.items()},
        "wilson_95_by_residue": {
            str(r): {"lower": interval[0], "upper": interval[1]} for r, interval in wilson95.items()
        },
        "absolute_accuracy_difference_r_vs_r_plus_4": {
            str(r): value for r, value in pair_differences.items()
        },
        "mean_absolute_accuracy_difference_r_vs_r_plus_4": sum(pair_differences.values()) / 4,
        "gap": max(accuracy.values()) - min(accuracy.values()),
    }


def default_leading_ids(tokenizer: Any) -> list[int]:
    plain = list(tokenizer.encode("a", add_special_tokens=False))
    full = list(tokenizer.encode("a", add_special_tokens=True))
    if not plain or full[-len(plain) :] != plain:
        raise ValueError("推理 Tokenizer 在文本末尾添加了 special token")
    return full[: len(full) - len(plain)]


def input_record(
    row: dict[str, Any],
    prompt_ids: Sequence[int],
    tokenizer: Any,
    leading_ids: Sequence[int],
) -> dict[str, Any]:
    metadata = row["metadata"]
    leading_count = len(leading_ids)
    if len(prompt_ids) != metadata["prompt_token_count"] + leading_count:
        raise ValueError(f"实际输入长度与 prompt 记录不一致：{row['sample_id']}")
    if list(prompt_ids[:leading_count]) != list(leading_ids):
        raise ValueError(f"实际输入的前导 special token 不正确：{row['sample_id']}")
    actual_hash = token_ids_sha256(prompt_ids)
    prompt_hash = token_ids_sha256(prompt_ids[leading_count:])
    expected_hash = metadata.get("input_token_ids_sha256")
    if expected_hash is not None and prompt_hash != expected_hash:
        raise ValueError(f"实际输入的 token ID 与构造记录不一致：{row['sample_id']}")
    key_position = metadata["target_pair_token_start"] + leading_count
    value_position = metadata["target_value_token_start"] + leading_count
    if (
        not 0 <= key_position < value_position < len(prompt_ids)
        or value_position != key_position + 2
    ):
        raise ValueError(f"目标 key、value 的位置不正确：{row['sample_id']}")
    key_id, value_id = int(prompt_ids[key_position]), int(prompt_ids[value_position])
    key = tokenizer.decode([key_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    value = tokenizer.decode(
        [value_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    query, answer = expected_query_and_answer(row, key, value)
    if row["answer"] != [answer]:
        raise ValueError(f"实际输入位置上的目标 token 与 gold 不一致：{row['sample_id']}")
    if row["components"]["query"] != query:
        raise ValueError(f"实际输入位置上的目标 token 与 query 不一致：{row['sample_id']}")
    return {
        "prompt_token_count": len(prompt_ids),
        "prompt_token_ids_sha256": actual_hash,
        "constructed_prompt_token_ids_sha256": prompt_hash,
        "leading_special_token_count": leading_count,
        "query_token_start": metadata["query_token_start"] + leading_count,
        "target_key": {"text": key, "token_id": key_id, "token_position": key_position},
        "target_value": {"text": value, "token_id": value_id, "token_position": value_position},
    }


def serialize_top10(
    token_ids: Sequence[int],
    logprobs: Sequence[float],
    ranks: Sequence[int | None],
    tokenizer: Any,
) -> list[dict[str, Any]]:
    candidates = []
    for token_id, logprob, rank in zip(token_ids, logprobs, ranks, strict=True):
        if rank is None or rank < 1:
            raise ValueError("top 10 的 vocabulary rank 不正确")
        if rank > TOP_LOGPROBS:
            continue
        logprob = float(logprob)
        if not math.isfinite(logprob) or logprob > 0:
            raise ValueError("top 10 的 raw logprob 必须是有限的非正数")
        token_id = int(token_id)
        candidates.append(
            {
                "rank": int(rank),
                "token_id": token_id,
                "vocabulary_token": tokenizer.convert_ids_to_tokens([token_id])[0],
                "decoded_text": tokenizer.decode(
                    [token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False
                ),
                "logprob": logprob,
                "probability": math.exp(logprob),
            }
        )
    candidates.sort(key=lambda item: item["rank"])
    if [item["rank"] for item in candidates] != list(range(1, TOP_LOGPROBS + 1)):
        raise ValueError("推理结果没有包含完整的 top 10")
    if len({item["token_id"] for item in candidates}) != TOP_LOGPROBS:
        raise ValueError("top 10 包含重复的 token ID")
    if any(left["logprob"] < right["logprob"] for left, right in pairwise(candidates)):
        raise ValueError("top 10 的概率顺序与 vocabulary rank 不一致")
    if sum(item["probability"] for item in candidates) > 1 + 1e-5:
        raise ValueError("top 10 的概率总和超过一")
    return candidates


def result_record(
    row: dict[str, Any], output: Any, tokenizer: Any, leading_ids: Sequence[int]
) -> dict[str, Any]:
    if not output.finished or len(output.outputs) != 1 or output.prompt_token_ids is None:
        raise ValueError(f"推理没有返回完整的单条结果：{row['sample_id']}")
    completion = output.outputs[0]
    if not completion.token_ids or completion.finish_reason is None:
        raise ValueError(f"推理没有返回生成 token 或结束原因：{row['sample_id']}")
    if completion.logprobs is None or len(completion.logprobs) != len(completion.token_ids):
        raise ValueError(f"生成 token 与 logprob 的数量不一致：{row['sample_id']}")
    generated_token_logprobs = []
    for token_id, position_logprobs in zip(completion.token_ids, completion.logprobs, strict=True):
        selected = position_logprobs.get(int(token_id))
        if selected is None or not math.isfinite(float(selected.logprob)):
            raise ValueError(f"生成 token 缺少有效 logprob：{row['sample_id']}")
        selected_logprob = float(selected.logprob)
        if selected_logprob > 0:
            raise ValueError(f"生成 token 的 logprob 大于零：{row['sample_id']}")
        generated_token_logprobs.append(
            {
                "token_id": int(token_id),
                "logprob": selected_logprob,
                "probability": math.exp(selected_logprob),
            }
        )
    # 第一个生成位置对应完整 prompt 最后一个输入位置的 next-token 分布。
    first_logprobs = completion.logprobs[0]
    top10 = serialize_top10(
        list(first_logprobs),
        [item.logprob for item in first_logprobs.values()],
        [item.rank for item in first_logprobs.values()],
        tokenizer,
    )
    input_info = input_record(row, output.prompt_token_ids, tokenizer, leading_ids)
    return {
        "sample_id": row["sample_id"],
        "request_id": output.request_id,
        "phase": row["phase"],
        "row_index": row["row_index"],
        "binding_id": row["binding_id"],
        "sentence_pair_id": row["sentence_pair_id"],
        "gold_answer": row["answer"][0],
        "input": input_info,
        "completion": {
            "text": completion.text,
            "token_ids": [int(token) for token in completion.token_ids],
            "token_logprobs": generated_token_logprobs,
            "cumulative_logprob": (
                None
                if completion.cumulative_logprob is None
                else float(completion.cumulative_logprob)
            ),
            "correct": is_correct(completion.text, row["answer"][0]),
            "finish_reason": completion.finish_reason,
            "stop_reason": completion.stop_reason,
        },
        "next_token_distribution": {
            "context_token_position": len(output.prompt_token_ids) - 1,
            "predicted_token_position": len(output.prompt_token_ids),
            "logprobs_mode": "raw_logprobs",
            "top10": top10,
        },
    }


def prepare_token_prompts(
    rows: Sequence[dict[str, Any]], tokenizer: Any, leading_ids: Sequence[int]
) -> list[dict[str, Any]]:
    prompts = []
    for row in rows:
        prompt_ids = list(tokenizer.encode(row["prompt"], add_special_tokens=True))
        input_record(row, prompt_ids, tokenizer, leading_ids)
        prompts.append({"prompt": row["prompt"], "prompt_token_ids": prompt_ids})
    return prompts


def _progress_record(index: int, record: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "record",
        "index": index,
        "sample_id": record["sample_id"],
        "phase": record["phase"],
        "correct": record["completion"]["correct"],
        "completion_text": record["completion"]["text"],
        "finished_at": time.time(),
        "record": record,
    }


def generate(
    model: str,
    rows: Sequence[dict[str, Any]],
    *,
    max_tokens: int = 8,
    batch_size: int = 16,
    progress_path: str | Path | None = None,
    progress_run_metadata: dict[str, Any] | None = None,
    **engine_args: Any,
) -> dict[str, Any]:

    from vllm import LLM, SamplingParams

    if batch_size <= 0:
        raise ValueError("batch_size 必须为正数")
    progress_file = None if progress_path is None else Path(progress_path)
    started_at = time.time()
    if progress_file is not None:
        if progress_file.exists():
            raise FileExistsError(f"评分进度文件已经存在：{progress_file}")
        progress_file.parent.mkdir(parents=True, exist_ok=True)

    sampling = {"temperature": 0, "max_tokens": max_tokens, "logprobs": TOP_LOGPROBS}
    sampling_params = SamplingParams(**sampling)
    versions = {
        name: importlib.metadata.version(name) for name in ("vllm", "torch", "transformers")
    }
    run_info: dict[str, Any] = {
        "model": model,
        "engine": "vllm",
        "versions": versions,
        "engine_args": {
            **engine_args,
            "generation_config": "vllm",
            "logprobs_mode": "raw_logprobs",
            "max_logprobs": TOP_LOGPROBS,
        },
        "sampling_params": sampling,
        "batch_size": batch_size,
        "progress_path": None
        if progress_file is None
        else str(progress_file.resolve()),
        "token_position_base": 0,
    }
    if progress_run_metadata is not None:
        run_info.update(progress_run_metadata)
    if progress_file is not None:
        with progress_file.open("w", encoding="utf-8") as progress:
            progress.write(
                json.dumps(
                    {
                        "type": "start",
                        "model": model,
                        "total_rows": len(rows),
                        "batch_size": batch_size,
                        "started_at": started_at,
                        "run": run_info,
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
    llm = LLM(
        model=model,
        generation_config="vllm",
        logprobs_mode="raw_logprobs",
        max_logprobs=TOP_LOGPROBS,
        **engine_args,
    )
    tokenizer = llm.get_tokenizer()
    leading_ids = default_leading_ids(tokenizer)
    records: list[dict[str, Any]] = []
    progress_context = (
        nullcontext()
        if progress_file is None
        else progress_file.open("a", encoding="utf-8")
    )
    with progress_context as progress:
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            prompts = prepare_token_prompts(batch_rows, tokenizer, leading_ids)
            outputs = llm.generate(prompts, sampling_params)
            batch_records = [
                result_record(row, output, tokenizer, leading_ids)
                for row, output in zip(batch_rows, outputs, strict=True)
            ]
            records.extend(batch_records)
            if progress is not None:
                for offset, record in enumerate(batch_records, start=start):
                    progress.write(
                        json.dumps(
                            _progress_record(offset, record),
                            ensure_ascii=False,
                            allow_nan=False,
                        )
                        + "\n"
                    )
                progress.flush()
        summary = summarize(rows, [record["completion"]["text"] for record in records])
        if progress is not None:
            progress.write(
                json.dumps(
                    {
                        "type": "complete",
                        "finished_at": time.time(),
                        "summary": summary,
                        "run": run_info,
                        "schema_version": 2,
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n"
            )
            progress.flush()
    return {
        "run": run_info,
        "summary": summary,
        "records": records,
    }
