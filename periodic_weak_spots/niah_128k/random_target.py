from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from periodic_weak_spots.tokenization import Tokenizer, token_ids, token_ids_sha256

from .config import ExperimentConfig, LayoutSpec
from .layout import (
    PAIR_FORM,
    PAIR_TOKENS,
    QUERY_MIDDLE,
    QUERY_PREFIX,
    QUERY_SUFFIX,
    VALUE_OFFSET,
    _guarded_padding,
    render_chat_prompt,
    render_query,
)
from .reverse_query import (
    REVERSE_QUERY_MIDDLE,
    REVERSE_QUERY_PREFIX,
    REVERSE_QUERY_SUFFIX,
    _tokenizer_for,
    render_reverse_query,
)

# 每个 binding 随机抽取一条 record 作为目标。目标 record 相对 record body 起点的偏移为 4t，
# 在 mod 8 下为 0 或 4；偏移为 4 时 body_prefix 缩短 4 个 token、tail 延长 4 个 token，
# 使目标 key 的相位仍由自然语言前句长度决定，prompt 总长度与 query 起点不变。
RANDOM_TARGET_SEED = 20261029
TARGET_PROTOCOL = "random_index"
REPLACED_COMPONENTS = ("body_prefix", "tail", "query")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _template_stable(
    tokenizer: Tokenizer, query: str, token_id: int, prefix: str, middle: str, suffix: str
) -> bool:
    if not query.endswith(suffix):
        suffix = suffix.rstrip()
    expected = [
        *token_ids(tokenizer, prefix),
        token_id,
        *token_ids(tokenizer, middle),
        token_id,
        *token_ids(tokenizer, suffix),
    ]
    return token_ids(tokenizer, query) == expected


def pair_supports_both_directions(tokenizer: Tokenizer, pair: dict[str, Any]) -> bool:
    """目标 key 在正向 query、目标 value 在反向 query 中都保持单个 token，含 rstrip 形式。"""

    key_id, value_id = int(pair["key_token_id"]), int(pair["value_token_id"])
    forward = render_query(pair["key"])
    reverse = render_reverse_query(pair["value"])
    checks = (
        (forward, key_id, QUERY_PREFIX, QUERY_MIDDLE, QUERY_SUFFIX),
        (forward.rstrip(), key_id, QUERY_PREFIX, QUERY_MIDDLE, QUERY_SUFFIX),
        (reverse, value_id, REVERSE_QUERY_PREFIX, REVERSE_QUERY_MIDDLE, REVERSE_QUERY_SUFFIX),
        (
            reverse.rstrip(),
            value_id,
            REVERSE_QUERY_PREFIX,
            REVERSE_QUERY_MIDDLE,
            REVERSE_QUERY_SUFFIX,
        ),
    )
    return all(_template_stable(tokenizer, *check) for check in checks)


def sample_target_indices(
    bindings: dict[int, dict[str, Any]], tokenizer: Tokenizer, *, pair_count: int, seed: int
) -> tuple[dict[int, int], int]:
    """按 binding_id 升序，为每个 binding 均匀抽取一个目标 record；返回目标与被拒绝的抽样次数。"""

    rng = random.Random(f"{seed}:random-target:{pair_count}")
    targets: dict[int, int] = {}
    rejected = 0
    for binding_id in sorted(bindings):
        pairs = bindings[binding_id]["pairs"]
        if len(pairs) != pair_count:
            raise ValueError(f"binding {binding_id} 的 record 数量必须为 {pair_count}")
        while True:
            index = rng.randrange(pair_count)
            if pair_supports_both_directions(tokenizer, pairs[index]):
                break
            rejected += 1
        targets[binding_id] = index
    return targets, rejected


def retarget_binding(binding: dict[str, Any], target_index: int) -> dict[str, Any]:
    pair = binding["pairs"][target_index]
    output = dict(binding)
    output["target_index"] = target_index
    output["target_key"] = pair["key"]
    output["target_value"] = pair["value"]
    return output


def phase_shift(target_index: int, alignment: int) -> int:
    return (target_index * PAIR_TOKENS) % alignment


def build_padding_variants(
    tokenizer: Tokenizer, reference_components: dict[str, str], filler: str, layout: LayoutSpec
) -> dict[int, tuple[str, str]]:
    """每种相位偏移对应一组 (body_prefix, tail)；偏移为 0 时沿用参考组件。"""

    body_prefix = reference_components["body_prefix"]
    tail = reference_components["tail"]
    body_count = len(token_ids(tokenizer, body_prefix))
    tail_count = len(token_ids(tokenizer, tail))
    body_guard = layout.body_prefix_guard_token
    tail_guard = layout.tail_guard_token

    def padding(count: int, guard: str, label: str) -> str:
        return _guarded_padding(tokenizer, count, filler, guard, label=label)

    if padding(body_count, body_guard, "body prefix") != body_prefix:
        raise ValueError("参考数据的 body_prefix 无法用给定 filler 重新构造")
    if padding(tail_count, tail_guard, "tail") != tail:
        raise ValueError("参考数据的 tail 无法用给定 filler 重新构造")
    shifts = sorted({phase_shift(index, layout.alignment) for index in range(layout.alignment)})
    variants: dict[int, tuple[str, str]] = {}
    for shift in shifts:
        if shift == 0:
            variants[shift] = (body_prefix, tail)
            continue
        variants[shift] = (
            padding(body_count - shift, body_guard, "shifted body prefix"),
            padding(tail_count + shift, tail_guard, "shifted tail"),
        )
    return variants


def _expected_span(
    label: str, position: int, reference: dict[str, int], *, body: int, tail: int, shift: int
) -> dict[str, int]:
    if position < body or position > tail:
        return dict(reference)
    if label == "body_prefix":
        return {"start": reference["start"], "count": reference["count"] - shift}
    if label == "tail":
        return {"start": reference["start"] - shift, "count": reference["count"] + shift}
    return {"start": reference["start"] - shift, "count": reference["count"]}


def _build_row(
    reference_row: dict[str, Any],
    binding: dict[str, Any],
    tokenizer: Tokenizer,
    *,
    variants: dict[int, tuple[str, str]],
    alignment: int,
) -> dict[str, Any]:
    target_index = int(binding["target_index"])
    target = binding["pairs"][target_index]
    key_id, value_id = int(target["key_token_id"]), int(target["value_token_id"])
    label = reference_row["sample_id"]
    reference_components = reference_row["components"]
    reference_metadata = reference_row["metadata"]
    reference_spans = reference_metadata["spans"]
    query = render_query(target["key"])
    if "chat_prefix" in reference_components:
        query = query.rstrip()
    shift = phase_shift(target_index, alignment)
    body_prefix, tail = variants[shift]
    replacements = {"body_prefix": body_prefix, "tail": tail, "query": query}
    # JSON 中的组件按名称排序，真实顺序由参考 span 的起点决定
    order = sorted(reference_components, key=lambda name: int(reference_spans[name]["start"]))
    components = {name: replacements.get(name, reference_components[name]) for name in order}
    prompt = "".join(components.values())
    component_ids = {name: token_ids(tokenizer, text) for name, text in components.items()}
    separate = [token for ids in component_ids.values() for token in ids]
    prompt_ids = token_ids(tokenizer, prompt)
    if prompt_ids != separate:
        raise ValueError(f"{label} 的组件边界无法独立编码")
    if len(prompt_ids) != int(reference_metadata["prompt_token_count"]):
        raise ValueError(f"{label} 的 prompt 长度与参考数据不一致")
    if "chat_prefix" in components:
        user_text = "".join(
            text for name, text in components.items() if name not in {"chat_prefix", "chat_suffix"}
        )
        if render_chat_prompt(tokenizer, user_text) != prompt:
            raise ValueError(f"{label} 与 Qwen non-thinking chat template 不一致")
    starts: dict[str, int] = {}
    cursor = 0
    for name, ids in component_ids.items():
        starts[name] = cursor
        cursor += len(ids)
    spans = {name: {"start": starts[name], "count": len(component_ids[name])} for name in order}
    body_position, tail_position = order.index("body_prefix"), order.index("tail")
    for position, name in enumerate(order):
        expected = _expected_span(
            name,
            position,
            reference_spans[name],
            body=body_position,
            tail=tail_position,
            shift=shift,
        )
        if spans[name] != expected:
            raise ValueError(f"{label} 的组件 {name} 位置或长度不正确")
    target_pair_start = starts["pair_body"] + target_index * PAIR_TOKENS
    target_value_start = target_pair_start + VALUE_OFFSET
    if target_pair_start % alignment != int(reference_row["phase"]):
        raise ValueError(f"{label} 的目标 phase 不正确")
    if prompt_ids[target_pair_start] != key_id or prompt_ids[target_value_start] != value_id:
        raise ValueError(f"{label} 的目标 key 或 value 位置不正确")
    if token_ids(tokenizer, prompt + target["value"]) != [*prompt_ids, value_id]:
        raise ValueError(f"{label} 的答案 value 无法追加")
    query_start = starts["query"]
    metadata = dict(reference_metadata)
    metadata.update(
        {
            "input_token_ids_sha256": token_ids_sha256(prompt_ids),
            "target_index": target_index,
            "target_pair_token_start": target_pair_start,
            "target_value_token_start": target_value_start,
            "target_phase": target_pair_start % alignment,
            "query_token_start": query_start,
            "query_token_count": len(component_ids["query"]),
            "target_to_query_distance": query_start - target_value_start,
            "target_protocol": TARGET_PROTOCOL,
            "body_prefix_shift_tokens": shift,
            "spans": spans,
        }
    )
    output = dict(reference_row)
    output["answer"] = [target["value"]]
    output["components"] = components
    output["prompt"] = prompt
    output["metadata"] = metadata
    return output


def _summary(values: list[int]) -> dict[str, float | int]:
    return {"min": min(values), "max": max(values), "mean": sum(values) / len(values)}


def _check_reference(metadata: dict[str, Any], filler: str) -> ExperimentConfig:
    if "target_protocol" in metadata:
        raise ValueError("参考数据已经是随机目标数据集")
    if "query_protocol" in metadata or "padding_protocol" in metadata:
        raise ValueError("参考数据必须是正向 query 的 Prefix Padding 数据集")
    if hashlib.sha256(filler.encode("utf-8")).hexdigest() != metadata["filler_sha256"]:
        raise ValueError("filler 与参考数据使用的 filler 不一致")
    return ExperimentConfig.from_dict(metadata["config"])


def build_dataset(
    reference_root: str | Path,
    output_root: str | Path,
    tokenizer: Tokenizer,
    *,
    filler: str,
    seed: int = RANDOM_TARGET_SEED,
    overwrite: bool = False,
) -> dict[str, Any]:
    reference = Path(reference_root)
    output = Path(output_root)
    if output.exists() and not overwrite:
        raise FileExistsError(f"输出目录已经存在：{output}")
    metadata = json.loads((reference / "metadata.json").read_text(encoding="utf-8"))
    layout = _check_reference(metadata, filler).layout
    reference_bindings = _read_jsonl(reference / "bindings.jsonl")
    by_id = {int(binding["binding_id"]): binding for binding in reference_bindings}
    if len(by_id) != len(reference_bindings):
        raise ValueError("binding_id 存在重复")
    targets, rejected = sample_target_indices(
        by_id, tokenizer, pair_count=layout.pair_count, seed=seed
    )
    bindings = {
        binding_id: retarget_binding(binding, targets[binding_id])
        for binding_id, binding in by_id.items()
    }
    phases = [int(phase) for phase in metadata["matrix"]["phases"]]
    first_row = _read_jsonl(reference / "rows" / PAIR_FORM / f"phase_{phases[0]}.jsonl")[0]
    variants = build_padding_variants(tokenizer, first_row["components"], filler, layout)
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        output / "bindings.jsonl",
        [bindings[int(binding["binding_id"])] for binding in reference_bindings],
    )
    target_starts: list[int] = []
    for phase in phases:
        reference_rows = _read_jsonl(reference / "rows" / PAIR_FORM / f"phase_{phase}.jsonl")
        rows = []
        for reference_row in reference_rows:
            row = _build_row(
                reference_row,
                bindings[int(reference_row["binding_id"])],
                tokenizer,
                variants=variants,
                alignment=layout.alignment,
            )
            target_starts.append(int(row["metadata"]["target_pair_token_start"]))
            rows.append(row)
        _write_jsonl(output / "rows" / PAIR_FORM / f"phase_{phase}.jsonl", rows)
    schedule = reference / "sentence_schedule.jsonl"
    if schedule.is_file():
        (output / schedule.name).write_bytes(schedule.read_bytes())
    indices = [targets[binding_id] for binding_id in sorted(targets)]
    output_metadata = dict(metadata)
    output_metadata["dataset_id"] = f"{metadata['dataset_id']}-random-target"
    output_metadata["target_protocol"] = {
        "mode": TARGET_PROTOCOL,
        "seed": seed,
        "pair_count": layout.pair_count,
        "rejected_draws": rejected,
        "phase_shift_tokens": sorted(variants),
        "shifted_binding_count": sum(
            1 for index in indices if phase_shift(index, layout.alignment)
        ),
        "target_index": _summary(indices),
        "target_pair_token_start": _summary(target_starts),
        "reference_target_index": layout.target_index,
        "reference_dataset": str(reference.resolve()),
    }
    output_metadata["design"] = {
        **metadata["design"],
        "target_position_moments": {
            PAIR_FORM: {
                **_summary(target_starts),
                "variable_term": "target_index * 4 + natural_prefill_token_count",
            }
        },
    }
    (output / "metadata.json").write_text(
        json.dumps(output_metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_metadata


def validate_dataset(
    dataset_root: str | Path,
    reference_root: str | Path,
    tokenizer: Tokenizer,
    *,
    filler: str,
) -> dict[str, Any]:
    """用参考数据、filler 与 metadata 中的 seed 逐行重新构造，要求与数据集逐字节相同。"""

    dataset = Path(dataset_root)
    reference = Path(reference_root)
    metadata = json.loads((dataset / "metadata.json").read_text(encoding="utf-8"))
    reference_metadata = json.loads((reference / "metadata.json").read_text(encoding="utf-8"))
    protocol = metadata.get("target_protocol")
    if not isinstance(protocol, dict) or protocol.get("mode") != TARGET_PROTOCOL:
        raise ValueError("metadata 中的随机目标参数不正确")
    if metadata["dataset_id"] != f"{reference_metadata['dataset_id']}-random-target":
        raise ValueError("dataset_id 与参考数据不对应")
    layout = _check_reference(reference_metadata, filler).layout
    if int(protocol["pair_count"]) != layout.pair_count:
        raise ValueError("metadata 中的 pair_count 与 layout 不一致")
    reference_bindings = _read_jsonl(reference / "bindings.jsonl")
    bindings = _read_jsonl(dataset / "bindings.jsonl")
    by_id = {int(binding["binding_id"]): binding for binding in reference_bindings}
    targets, rejected = sample_target_indices(
        by_id, tokenizer, pair_count=layout.pair_count, seed=int(protocol["seed"])
    )
    if rejected != int(protocol["rejected_draws"]):
        raise ValueError("被拒绝的抽样次数与 metadata 不一致")
    expected_bindings = [
        retarget_binding(binding, targets[int(binding["binding_id"])])
        for binding in reference_bindings
    ]
    if bindings != expected_bindings:
        raise ValueError("bindings.jsonl 与按 seed 重新抽取的目标不一致")
    schedule = reference / "sentence_schedule.jsonl"
    if schedule.is_file() and (dataset / schedule.name).read_bytes() != schedule.read_bytes():
        raise ValueError("sentence_schedule.jsonl 与参考数据不一致")
    by_target = {int(binding["binding_id"]): binding for binding in expected_bindings}
    phases = [int(phase) for phase in metadata["matrix"]["phases"]]
    first_row = _read_jsonl(reference / "rows" / PAIR_FORM / f"phase_{phases[0]}.jsonl")[0]
    variants = build_padding_variants(tokenizer, first_row["components"], filler, layout)
    query_starts: set[int] = set()
    target_starts: list[int] = []
    checked = 0
    for phase in phases:
        rows = _read_jsonl(dataset / "rows" / PAIR_FORM / f"phase_{phase}.jsonl")
        reference_rows = _read_jsonl(reference / "rows" / PAIR_FORM / f"phase_{phase}.jsonl")
        if len(rows) != len(reference_rows):
            raise ValueError(f"phase {phase} 的行数与参考数据不一致")
        for row, reference_row in zip(rows, reference_rows, strict=True):
            expected = _build_row(
                reference_row,
                by_target[int(reference_row["binding_id"])],
                tokenizer,
                variants=variants,
                alignment=layout.alignment,
            )
            if row != expected:
                raise ValueError(f"{reference_row['sample_id']} 与重新构造的结果不一致")
            query_starts.add(int(row["metadata"]["query_token_start"]))
            target_starts.append(int(row["metadata"]["target_pair_token_start"]))
            checked += 1
    if len(query_starts) != 1:
        raise ValueError("query 起点随样本变化")
    if checked != int(metadata["matrix"]["total_rows"]):
        raise ValueError("检查的行数与 metadata 不一致")
    if protocol["target_pair_token_start"] != _summary(target_starts):
        raise ValueError("metadata 中的目标位置统计不正确")
    return {
        "status": "ok",
        "dataset_id": metadata["dataset_id"],
        "validated_rows": checked,
        "target_protocol": protocol,
        "query_start": query_starts.pop(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m periodic_weak_spots.niah_128k.random_target")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="从 Prefix Padding 参考数据集构造随机目标数据集")
    build.add_argument("--reference-dataset", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--tokenizer-path", type=Path, required=True)
    build.add_argument("--filler-file", type=Path, required=True, help="与参考数据相同的 filler")
    build.add_argument("--seed", type=int, default=RANDOM_TARGET_SEED)
    build.add_argument("--overwrite", action="store_true")
    validate = commands.add_parser("validate", help="用参考数据与 seed 逐行重新构造并比较")
    validate.add_argument("--dataset", type=Path, required=True)
    validate.add_argument("--reference-dataset", type=Path, required=True)
    validate.add_argument("--tokenizer-path", type=Path, required=True)
    validate.add_argument("--filler-file", type=Path, required=True)
    args = parser.parse_args()
    filler = args.filler_file.read_bytes().decode("utf-8")
    if args.command == "build":
        tokenizer = _tokenizer_for(args.reference_dataset / "metadata.json", args.tokenizer_path)
        metadata = build_dataset(
            args.reference_dataset,
            args.output,
            tokenizer,
            filler=filler,
            seed=args.seed,
            overwrite=args.overwrite,
        )
        print(json.dumps(metadata["target_protocol"], ensure_ascii=False, indent=2, sort_keys=True))
        return
    tokenizer = _tokenizer_for(args.dataset / "metadata.json", args.tokenizer_path)
    report = validate_dataset(args.dataset, args.reference_dataset, tokenizer, filler=filler)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
