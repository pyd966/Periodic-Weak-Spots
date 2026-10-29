from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from functools import cache
from pathlib import Path
from typing import Any

from periodic_weak_spots.tokenization import load_tokenizer, token_ids, token_ids_sha256

from .layout import PAIR_FORM, PAIR_TOKENS, VALUE_OFFSET, render_query
from .schedule import BALANCED_BLOCK_ROWS, phase_lengths

PAD_TEXT = "｜"
PAD_TOKEN_ID = 158819
OUTER_GUARD_TOKENS = 8
TOTAL_GAP_EXTRAS = 32
MAX_GAP_EXTRAS = 4
REFERENCE_FILLER_OFFSET = 8
GAP_SEED = 20260721


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _load_bindings(root: Path) -> dict[int, dict[str, Any]]:
    bindings = _read_jsonl(root / "bindings.jsonl")
    result = {int(row["binding_id"]): row for row in bindings}
    if len(result) != len(bindings):
        raise ValueError("binding_id 存在重复")
    return result


def _phase_reference_lengths(phase: int, rows_per_phase: int) -> list[int]:
    if rows_per_phase <= 0 or rows_per_phase % BALANCED_BLOCK_ROWS:
        raise ValueError("rows_per_phase 必须是256的正整数倍")
    lengths = phase_lengths(phase)
    return lengths * (rows_per_phase // BALANCED_BLOCK_ROWS)


@cache
def _composition_table(parts: int, total: int, cap: int) -> tuple[tuple[int, ...], ...]:
    if parts <= 0 or total < 0 or cap < 0:
        raise ValueError("bounded composition 参数不正确")
    table: list[list[int]] = [[0] * (total + 1) for _ in range(parts + 1)]
    table[0][0] = 1
    for count in range(1, parts + 1):
        previous = table[count - 1]
        current = table[count]
        window = 0
        for value in range(total + 1):
            window += previous[value]
            expired = value - cap - 1
            if expired >= 0:
                window -= previous[expired]
            current[value] = window
    return tuple(tuple(row) for row in table)


def _sample_bounded_composition(
    *,
    total: int,
    parts: int,
    cap: int,
    rng: random.Random,
) -> tuple[int, ...]:
    if total < 0 or parts <= 0 or cap < 0:
        raise ValueError("bounded composition 参数不正确")
    table = _composition_table(parts, total, cap)
    if table[parts][total] == 0:
        raise ValueError("bounded composition 无法满足总长度和上限")
    result: list[int] = []
    remaining = total
    for index in range(parts):
        remaining_parts = parts - index - 1
        weighted_values = tuple(
            (
                value,
                table[remaining_parts][remaining - value]
                if 0 <= remaining - value <= total
                else 0,
            )
            for value in range(cap + 1)
            if 0 <= remaining - value <= total
            and table[remaining_parts][remaining - value]
        )
        draw = rng.randrange(sum(weight for _, weight in weighted_values))
        cumulative = 0
        for value, weight in weighted_values:
            cumulative += weight
            if draw < cumulative:
                result.append(value)
                remaining -= value
                break
    if remaining or len(result) != parts:
        raise RuntimeError("bounded composition 生成失败")
    return tuple(result)


def _gap_assignment(
    *,
    phase: int,
    row_index: int,
    reference_length: int,
    pair_count: int,
    target_index: int,
) -> tuple[int, tuple[int, ...]]:
    before_total = reference_length - REFERENCE_FILLER_OFFSET
    after_total = TOTAL_GAP_EXTRAS - before_total
    gap_rng = random.Random(
        f"{GAP_SEED}:in-sequence-v1:{phase}:{row_index}:{pair_count}:{target_index}"
    )
    before = _sample_bounded_composition(
        total=before_total,
        parts=target_index,
        cap=MAX_GAP_EXTRAS,
        rng=gap_rng,
    )
    after = _sample_bounded_composition(
        total=after_total,
        parts=pair_count - target_index - 1,
        cap=MAX_GAP_EXTRAS,
        rng=gap_rng,
    )
    gaps = (*before, *after)
    if len(gaps) != pair_count - 1 or sum(gaps) != TOTAL_GAP_EXTRAS:
        raise RuntimeError("gap schedule 的长度不正确")
    return reference_length, gaps


def _body_records(binding: dict[str, Any], record_separator: str) -> list[str]:
    return [
        f"{pair['key']}:{pair['value']}{record_separator}"
        for pair in binding["pairs"]
    ]


def _padded_body(records: list[str], gaps: tuple[int, ...]) -> str:
    if len(gaps) != len(records) - 1:
        raise ValueError("gap 数量与 record 数量不一致")
    return "".join(
        record + (PAD_TEXT * gaps[index] if index < len(gaps) else "")
        for index, record in enumerate(records)
    )


def _split_body_records(body: str) -> list[str]:
    pieces = body.split("。")
    if not pieces or pieces[-1] != "":
        raise ValueError("pair_body 没有以 record separator 结束")
    return [piece + "。" for piece in pieces[:-1]]


def _component_order(use_chat_template: bool) -> tuple[str, ...]:
    order = (
        "intro",
        "in_sequence_prefix_guard",
        "body_prefix",
        "pair_body",
        "in_sequence_suffix_guard",
        "tail",
        "query_separator",
        "query",
    )
    return ("chat_prefix", *order, "chat_suffix") if use_chat_template else order


def _fixed_components(reference_row: dict[str, Any], use_chat_template: bool) -> dict[str, str]:
    components = reference_row["components"]
    required = {
        "intro",
        "body_prefix",
        "tail",
        "query_separator",
        "chat_prefix",
        "chat_suffix",
    }
    if use_chat_template and not required.issubset(components):
        raise ValueError("参考数据缺少 Qwen chat template 组件")
    if not use_chat_template and not {"intro", "body_prefix", "tail", "query_separator"}.issubset(
        components
    ):
        raise ValueError("参考数据缺少固定 prompt 组件")
    result = {
        "intro": components["intro"],
        "body_prefix": components["body_prefix"],
        "tail": components["tail"],
        "query_separator": components["query_separator"],
    }
    if use_chat_template:
        result.update(
            {
                "chat_prefix": components["chat_prefix"],
                "chat_suffix": components["chat_suffix"],
            }
        )
    return result


def _assert_chat_template(tokenizer: Any, components: dict[str, str], prompt: str) -> None:
    order = _component_order("chat_prefix" in components)
    user_text = "".join(
        components[label]
        for label in order
        if label not in {"chat_prefix", "chat_suffix"}
    )
    rendered = tokenizer.hf.apply_chat_template(
        [{"role": "user", "content": user_text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if rendered != prompt:
        raise ValueError("In-sequence prompt 与 Qwen non-thinking chat template 不一致")


def _build_row(
    reference_row: dict[str, Any],
    binding: dict[str, Any],
    fixed: dict[str, str],
    base_body: str,
    tokenizer: Any,
    *,
    reference_length: int,
    gaps: tuple[int, ...],
    padding_metadata: dict[str, Any],
) -> dict[str, Any]:
    target_index = int(binding["target_index"])
    query = render_query(binding["target_key"]).rstrip()
    guard = PAD_TEXT * OUTER_GUARD_TOKENS
    components: dict[str, str] = {}
    if "chat_prefix" in fixed:
        components["chat_prefix"] = fixed["chat_prefix"]
    components.update(
        {
            "intro": fixed["intro"],
            "in_sequence_prefix_guard": guard,
            "body_prefix": fixed["body_prefix"],
            "pair_body": _padded_body(_split_body_records(base_body), gaps),
            "in_sequence_suffix_guard": guard,
            "tail": fixed["tail"],
            "query_separator": fixed["query_separator"],
            "query": query,
        }
    )
    if "chat_suffix" in fixed:
        components["chat_suffix"] = fixed["chat_suffix"]
    order = _component_order("chat_prefix" in fixed)
    if tuple(components) != order:
        raise AssertionError("组件顺序不正确")
    prompt = "".join(components[label] for label in order)
    component_ids = {label: token_ids(tokenizer, components[label]) for label in order}
    separate_ids = [token for label in order for token in component_ids[label]]
    prompt_ids = token_ids(tokenizer, prompt)
    if prompt_ids != separate_ids:
        raise ValueError(f"{reference_row['sample_id']} 的组件边界无法独立编码")
    if len(prompt_ids) != 128_000:
        raise ValueError(f"{reference_row['sample_id']} 的 prompt 长度必须为128000")
    if token_ids(tokenizer, prompt + binding["target_value"]) != [
        *prompt_ids,
        int(binding["pairs"][target_index]["value_token_id"]),
    ]:
        raise ValueError(f"{reference_row['sample_id']} 的答案 token 无法追加")
    if "chat_prefix" in fixed:
        _assert_chat_template(tokenizer, components, prompt)

    starts: dict[str, int] = {}
    cursor = 0
    for label in order:
        starts[label] = cursor
        cursor += len(component_ids[label])
    if cursor != len(prompt_ids):
        raise AssertionError("组件 span 没有填满 prompt")
    pair_start = starts["pair_body"]
    target_pair_start = pair_start + target_index * PAIR_TOKENS + sum(gaps[:target_index])
    target_value_start = target_pair_start + VALUE_OFFSET
    query_start = starts["query"]
    if target_pair_start % 8 != int(reference_row["phase"]):
        raise ValueError(f"{reference_row['sample_id']} 的目标 phase 不正确")
    if query_start != int(reference_row["metadata"]["query_token_start"]):
        raise ValueError(f"{reference_row['sample_id']} 的 query 起点发生变化")
    if target_pair_start != int(reference_row["metadata"]["target_pair_token_start"]):
        raise ValueError(f"{reference_row['sample_id']} 的目标位置发生变化")
    spans = {
        label: {"start": starts[label], "count": len(component_ids[label])}
        for label in order
    }
    metadata = {
        "input_token_ids_sha256": token_ids_sha256(prompt_ids),
        "prompt_token_count": len(prompt_ids),
        "model_prefill_token_count": len(prompt_ids),
        "phase_control_prefix_count": reference_length,
        "fixed_natural_filler_count": 48,
        "pair_token_count": PAIR_TOKENS,
        "target_index": target_index,
        "target_pair_token_start": target_pair_start,
        "target_value_token_start": target_value_start,
        "target_phase": target_pair_start % 8,
        "query_token_start": query_start,
        "query_token_count": len(component_ids["query"]),
        "target_to_query_distance": query_start - target_value_start,
        "padding_protocol": "in_sequence",
        **padding_metadata,
        "reference_prefill_token_count": reference_length,
        "gap_extra_lengths": list(gaps),
        "spans": spans,
    }
    output = dict(reference_row)
    output["answer"] = [binding["target_value"]]
    output["components"] = components
    output["prompt"] = prompt
    output["metadata"] = metadata
    output["prefill_sentence_id"] = "in-sequence-prefix-guard"
    output["postfill_sentence_id"] = "in-sequence-suffix-guard"
    return output


def _padding_metadata() -> dict[str, Any]:
    return {
        "pad_text": PAD_TEXT,
        "pad_token_id": PAD_TOKEN_ID,
        "outer_guard_tokens": OUTER_GUARD_TOKENS,
        "total_gap_extras": TOTAL_GAP_EXTRAS,
        "max_gap_extras": MAX_GAP_EXTRAS,
        "reference_filler_offset": REFERENCE_FILLER_OFFSET,
        "gap_seed": GAP_SEED,
    }


def build_dataset(
    reference_root: str | Path,
    output_root: str | Path,
    tokenizer: Any,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    reference = Path(reference_root)
    output = Path(output_root)
    if output.exists() and not overwrite:
        raise FileExistsError(f"输出目录已经存在：{output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "rows" / PAIR_FORM).mkdir(parents=True, exist_ok=True)
    metadata = json.loads((reference / "metadata.json").read_text(encoding="utf-8"))
    config = metadata["config"]
    layout = config["layout"]
    rows_per_phase = int(metadata["matrix"]["rows_per_cell"])
    pair_count = int(layout["pair_count"])
    target_index = int(layout["target_index"])
    bindings = _load_bindings(reference)
    if len(bindings) != rows_per_phase:
        raise ValueError("参考数据的 binding 数量不符合 phase 设计")
    reference_phase0 = _read_jsonl(reference / "rows" / PAIR_FORM / "phase_0.jsonl")
    if len(reference_phase0) != rows_per_phase:
        raise ValueError("参考数据 phase_0 行数不正确")
    fixed = _fixed_components(reference_phase0[0], True)
    body_by_binding: dict[int, str] = {}
    for row in reference_phase0:
        binding_id = int(row["binding_id"])
        binding = bindings[binding_id]
        records = _body_records(binding, layout["record_separator"])
        if len(records) != pair_count:
            raise ValueError("binding record 数量不正确")
        if "｜" in "".join(records):
            raise ValueError("binding record 不应包含 padding token")
        if row["components"]["pair_body"] != "".join(records):
            raise ValueError(f"参考数据 binding {binding_id} 的 pair_body 不一致")
        body_by_binding[binding_id] = row["components"]["pair_body"]
    pad_ids = token_ids(tokenizer, PAD_TEXT)
    if pad_ids != [PAD_TOKEN_ID]:
        raise ValueError(
            f"Qwen Tokenizer 的 `｜` token ID 为 {pad_ids}，预期为 {PAD_TOKEN_ID}"
        )
    for count in range(MAX_GAP_EXTRAS + 1):
        if token_ids(tokenizer, PAD_TEXT * count) != [PAD_TOKEN_ID] * count:
            raise ValueError("padding token 的重复编码不具备 token additivity")
    if token_ids(tokenizer, PAD_TEXT * OUTER_GUARD_TOKENS) != [PAD_TOKEN_ID] * OUTER_GUARD_TOKENS:
        raise ValueError("guard padding 的 token 数量不正确")

    padding = _padding_metadata()
    histogram: dict[str, Counter[int]] = {str(phase): Counter() for phase in range(8)}
    schedule_path = output / "padding_schedule.jsonl"
    with schedule_path.open("w", encoding="utf-8") as schedule_handle:
        for phase in range(8):
            reference_path = reference / "rows" / PAIR_FORM / f"phase_{phase}.jsonl"
            reference_rows = _read_jsonl(reference_path)
            if len(reference_rows) != rows_per_phase:
                raise ValueError(f"参考数据 phase {phase} 行数不正确")
            output_rows: list[dict[str, Any]] = []
            for reference_row in reference_rows:
                row_index = int(reference_row["row_index"])
                binding_id = int(reference_row["binding_id"])
                reference_length, gaps = _gap_assignment(
                    phase=phase,
                    row_index=row_index,
                    reference_length=int(
                        reference_row["metadata"]["phase_control_prefix_count"]
                    ),
                    pair_count=pair_count,
                    target_index=target_index,
                )
                row = _build_row(
                    reference_row,
                    bindings[binding_id],
                    fixed,
                    body_by_binding[binding_id],
                    tokenizer,
                    reference_length=reference_length,
                    gaps=gaps,
                    padding_metadata=padding,
                )
                output_rows.append(row)
                histogram[str(phase)][reference_length] += 1
                schedule_handle.write(
                    json.dumps(
                        {
                            "phase": phase,
                            "row_index": row_index,
                            "binding_id": binding_id,
                            "reference_prefill_token_count": reference_length,
                            "gap_extra_lengths": list(gaps),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
            _write_jsonl(output / "rows" / PAIR_FORM / f"phase_{phase}.jsonl", output_rows)

    output_metadata = dict(metadata)
    output_metadata["dataset_id"] = (
        f"{metadata['dataset_id']}-in-sequence"
    )
    output_metadata["padding_protocol"] = {
        "mode": "in_sequence",
        **padding,
        "reference_dataset": str(reference.resolve()),
    }
    output_metadata["source_report"] = {
        "mode": "in_sequence",
        "sentence_source": None,
        "reference_dataset": str(reference.resolve()),
    }
    output_metadata.pop("sentence_schedule_path", None)
    output_metadata["padding_schedule_path"] = "padding_schedule.jsonl"
    output_metadata["in_sequence_histogram"] = {
        phase: {str(length): count for length, count in sorted(values.items())}
        for phase, values in histogram.items()
    }
    (output / "bindings.jsonl").write_bytes((reference / "bindings.jsonl").read_bytes())
    (output / "metadata.json").write_text(
        json.dumps(output_metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_metadata


def _expected_body_ids(
    binding: dict[str, Any], tokenizer: Any, layout: dict[str, Any]
) -> str:
    records = _body_records(binding, layout["record_separator"])
    colon = token_ids(tokenizer, ":")
    separator = token_ids(tokenizer, layout["record_separator"])
    if len(colon) != 1 or len(separator) != 1:
        raise ValueError("record punctuation 必须分别占用一个 token")
    for pair, record in zip(binding["pairs"], records, strict=True):
        encoded = token_ids(tokenizer, record)
        expected = [
            int(pair["key_token_id"]),
            colon[0],
            int(pair["value_token_id"]),
            separator[0],
        ]
        if encoded != expected:
            raise ValueError(f"binding {binding['binding_id']} 的 record 编码不正确")
    return "".join(records)


def validate_dataset(root: str | Path, tokenizer: Any) -> dict[str, Any]:
    dataset = Path(root)
    metadata = json.loads((dataset / "metadata.json").read_text(encoding="utf-8"))
    protocol = metadata.get("padding_protocol")
    expected_protocol = {"mode": "in_sequence", **_padding_metadata()}
    if not isinstance(protocol, dict) or any(
        protocol.get(key) != value for key, value in expected_protocol.items()
    ):
        raise ValueError("metadata 中的 In-sequence Padding 参数不正确")
    config = metadata["config"]
    layout = config["layout"]
    rows_per_phase = int(metadata["matrix"]["rows_per_cell"])
    pair_count = int(layout["pair_count"])
    target_index = int(layout["target_index"])
    bindings = _load_bindings(dataset)
    schedule_handle = (dataset / metadata["padding_schedule_path"]).open(encoding="utf-8")
    pad_ids = token_ids(tokenizer, PAD_TEXT)
    if pad_ids != [PAD_TOKEN_ID]:
        raise ValueError("Qwen Tokenizer 的 padding token ID 不正确")
    for count in range(MAX_GAP_EXTRAS + 1):
        if token_ids(tokenizer, PAD_TEXT * count) != [PAD_TOKEN_ID] * count:
            raise ValueError("padding token 的重复编码不正确")
    body_cache: dict[int, str] = {}
    for binding_id, binding in bindings.items():
        body_cache[binding_id] = _expected_body_ids(binding, tokenizer, layout)
    order = _component_order(True)
    histograms: dict[int, Counter[int]] = {phase: Counter() for phase in range(8)}
    query_starts: set[int] = set()
    prompt_lengths: set[int] = set()
    checked = 0
    for phase in range(8):
        path = dataset / "rows" / PAIR_FORM / f"phase_{phase}.jsonl"
        rows = _read_jsonl(path)
        if len(rows) != rows_per_phase:
            raise ValueError(f"phase {phase} 行数不正确")
        for row in rows:
            row_index = int(row["row_index"])
            binding_id = int(row["binding_id"])
            binding = bindings[binding_id]
            schedule_line = schedule_handle.readline()
            if not schedule_line:
                raise ValueError("padding schedule 行数不足")
            assignment = json.loads(schedule_line)
            if (int(assignment["phase"]), int(assignment["row_index"])) != (
                phase,
                row_index,
            ):
                raise ValueError("padding schedule 坐标与 row 顺序不一致")
            reference_length = int(assignment["reference_prefill_token_count"])
            gaps = tuple(int(value) for value in assignment["gap_extra_lengths"])
            if reference_length % 8 != phase or not 8 <= reference_length <= 40:
                raise ValueError(f"phase {phase} row {row_index} 的参考长度不正确")
            if len(gaps) != pair_count - 1 or sum(gaps) != TOTAL_GAP_EXTRAS:
                raise ValueError(f"phase {phase} row {row_index} 的 gap 总量不正确")
            if any(value < 0 or value > MAX_GAP_EXTRAS for value in gaps):
                raise ValueError(f"phase {phase} row {row_index} 的 gap 超出范围")
            if tuple(row["metadata"]["gap_extra_lengths"]) != gaps:
                raise ValueError("row metadata 与 padding schedule 不一致")
            components = row["components"]
            if set(components) != set(order):
                raise ValueError("组件顺序不正确")
            if components["in_sequence_prefix_guard"] != PAD_TEXT * OUTER_GUARD_TOKENS:
                raise ValueError("前部 guard 内容不正确")
            if components["in_sequence_suffix_guard"] != PAD_TEXT * OUTER_GUARD_TOKENS:
                raise ValueError("后部 guard 内容不正确")
            base_body = body_cache[binding_id]
            expected_body = _padded_body(_split_body_records(base_body), gaps)
            if components["pair_body"] != expected_body:
                raise ValueError("pair_body 中的 gap 位置不正确")
            expected_query = render_query(binding["target_key"]).rstrip()
            if components["query"] != expected_query:
                raise ValueError("query 与 target key 不一致")
            prompt = "".join(components[label] for label in order)
            if prompt != row["prompt"]:
                raise ValueError("prompt 与组件连接结果不一致")
            component_ids = {label: token_ids(tokenizer, components[label]) for label in order}
            prompt_ids = token_ids(tokenizer, prompt)
            separate_ids = [token for label in order for token in component_ids[label]]
            if prompt_ids != separate_ids or len(prompt_ids) != 128_000:
                raise ValueError(f"phase {phase} row {row_index} 的 prompt token 不正确")
            _assert_chat_template(tokenizer, components, prompt)
            starts: dict[str, int] = {}
            cursor = 0
            for label in order:
                starts[label] = cursor
                cursor += len(component_ids[label])
            spans = row["metadata"]["spans"]
            expected_spans = {
                label: {"start": starts[label], "count": len(component_ids[label])}
                for label in order
            }
            if spans != expected_spans:
                raise ValueError("metadata span 与重新编码结果不一致")
            target_pair_start = (
                starts["pair_body"] + target_index * PAIR_TOKENS + sum(gaps[:target_index])
            )
            target_value_start = target_pair_start + VALUE_OFFSET
            if row["metadata"]["target_pair_token_start"] != target_pair_start:
                raise ValueError("目标 key 位置不正确")
            if row["metadata"]["target_value_token_start"] != target_value_start:
                raise ValueError("目标 value 位置不正确")
            if target_pair_start % 8 != phase:
                raise ValueError("目标 phase 不正确")
            if row["metadata"]["query_token_start"] != starts["query"]:
                raise ValueError("query 起点不正确")
            if row["metadata"]["reference_prefill_token_count"] != reference_length:
                raise ValueError("reference prefill length 不正确")
            if token_ids(tokenizer, prompt + binding["target_value"]) != [
                *prompt_ids,
                int(binding["pairs"][target_index]["value_token_id"]),
            ]:
                raise ValueError("答案 token 无法追加到 prompt")
            if row["metadata"]["input_token_ids_sha256"] != token_ids_sha256(prompt_ids):
                raise ValueError("prompt token hash 不正确")
            histograms[phase][reference_length] += 1
            query_starts.add(starts["query"])
            prompt_lengths.add(len(prompt_ids))
            checked += 1
    expected_histograms = {
        phase: Counter(_phase_reference_lengths(phase, rows_per_phase)) for phase in range(8)
    }
    if schedule_handle.readline():
        raise ValueError("padding schedule 行数超过 row 数量")
    schedule_handle.close()
    if histograms != expected_histograms:
        raise ValueError("phase reference length histogram 不正确")
    if query_starts != {127958} or prompt_lengths != {128_000}:
        raise ValueError("固定 query 起点或 prompt 长度不正确")
    return {
        "status": "ok",
        "dataset_id": metadata["dataset_id"],
        "validated_rows": checked,
        "padding_protocol": protocol,
        "query_start": 127958,
        "prompt_tokens": 128_000,
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m periodic_weak_spots.niah_128k.in_sequence")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--reference-dataset", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--tokenizer-path", type=Path, required=True)
    build.add_argument("--overwrite", action="store_true")
    validate = commands.add_parser("validate")
    validate.add_argument("--dataset", type=Path, required=True)
    validate.add_argument("--tokenizer-path", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        tokenizer = load_tokenizer(
            "Qwen/Qwen3.8-Flash-Next",
            "de4b8e4d43b917e7706784d8bb445c9af86a3540",
            local_path=args.tokenizer_path,
        )
        metadata = build_dataset(
            args.reference_dataset,
            args.output,
            tokenizer,
            overwrite=args.overwrite,
        )
        print(
            json.dumps(
                metadata["padding_protocol"],
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return
    tokenizer = load_tokenizer(
        "Qwen/Qwen3.8-Flash-Next",
        "de4b8e4d43b917e7706784d8bb445c9af86a3540",
        local_path=args.tokenizer_path,
    )
    report = validate_dataset(args.dataset, tokenizer)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
