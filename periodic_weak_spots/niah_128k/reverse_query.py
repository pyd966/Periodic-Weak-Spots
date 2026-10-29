from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from periodic_weak_spots.tokenization import (
    FastTokenizer,
    Tokenizer,
    build_token_exact_padding,
    load_tokenizer,
    token_ids,
    token_ids_sha256,
)

from .config import ExperimentConfig, LayoutSpec
from .layout import PAIR_FORM, _guarded_padding, render_chat_prompt, render_pair

# 反向 query：给出 value，要求回答对应的 key；三段模板的 token 数量与正向模板一致
REVERSE_QUERY_PREFIX = (
    "Now answer the question using the key-value pairs above. What is the key for value "
)
REVERSE_QUERY_MIDDLE = "? Answer with only the key. The key for value "
REVERSE_QUERY_SUFFIX = " is "
QUERY_DIRECTION = "value_to_key"
REPLACED_COMPONENTS = ("intro", "tail", "query")


def render_reverse_query(value: str) -> str:
    return REVERSE_QUERY_PREFIX + value + REVERSE_QUERY_MIDDLE + value + REVERSE_QUERY_SUFFIX


def render_reverse_intro(spec: LayoutSpec) -> str:
    examples = (("猫", "树"), ("狗", "叶"), ("鸟", "根"))
    rendered = ", ".join(render_pair(key, value, spec) for key, value in examples)
    return (
        "Some special key-value pairs are hidden within the following text. "
        "Make sure to memorize them. I will give you a value afterwards and ask for its key. "
        'Each key-value pair is written in the form key:value. The token before ":" '
        'is the key, and the token after ":" is the value.\n\n'
        f"For example, the text may contain {rendered}. "
        "These examples only explain the format.\n"
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _load_bindings(root: Path) -> dict[int, dict[str, Any]]:
    bindings = _read_jsonl(root / "bindings.jsonl")
    result = {int(row["binding_id"]): row for row in bindings}
    if len(result) != len(bindings):
        raise ValueError("binding_id 存在重复")
    return result


def _target_pair(binding: dict[str, Any]) -> dict[str, Any]:
    return binding["pairs"][int(binding["target_index"])]


def _query_text(value: str, use_chat_template: bool) -> str:
    query = render_reverse_query(value)
    return query.rstrip() if use_chat_template else query


def _assert_query_stable(tokenizer: Tokenizer, query: str, value_token_id: int, label: str) -> None:
    # value 在 query 中出现两次，两处都必须保持单个 token
    suffix = (
        REVERSE_QUERY_SUFFIX
        if query.endswith(REVERSE_QUERY_SUFFIX)
        else REVERSE_QUERY_SUFFIX.rstrip()
    )
    expected = [
        *token_ids(tokenizer, REVERSE_QUERY_PREFIX),
        value_token_id,
        *token_ids(tokenizer, REVERSE_QUERY_MIDDLE),
        value_token_id,
        *token_ids(tokenizer, suffix),
    ]
    if token_ids(tokenizer, query) != expected:
        raise ValueError(f"{label} 的 value 在反向 query 中必须保持单个 token")


def _assert_chat_template(tokenizer: Tokenizer, components: dict[str, str], prompt: str) -> None:
    if "chat_prefix" not in components:
        return
    user_text = "".join(
        text for label, text in components.items() if label not in {"chat_prefix", "chat_suffix"}
    )
    if render_chat_prompt(tokenizer, user_text) != prompt:
        raise ValueError("反向 query prompt 与 Qwen non-thinking chat template 不一致")


def build_reverse_intro(
    tokenizer: Tokenizer, reference_intro: str, filler: str, spec: LayoutSpec
) -> str:
    """反向 intro 用相同的 filler 补齐到参考 intro 的 token 数量。"""

    target_count = len(token_ids(tokenizer, reference_intro))
    base = render_reverse_intro(spec)
    for padding_count in range(target_count + 1):
        padding = build_token_exact_padding(tokenizer, padding_count, filler)
        candidate = base + padding + "\n"
        count = len(token_ids(tokenizer, candidate))
        if count == target_count:
            return candidate
        if count > target_count:
            break
    raise ValueError(f"反向 intro 无法补齐到参考 intro 的 {target_count} 个 token")


def build_tail(
    tokenizer: Tokenizer,
    reference_tail: str,
    filler: str,
    guard: str,
    *,
    reference_query_count: int,
    query_count: int,
) -> str:
    """query 的 token 数量变化时由 tail 补偿，使 prompt 总长度与 query 之前的位置不变。"""

    reference_count = len(token_ids(tokenizer, reference_tail))
    tail_count = reference_count + reference_query_count - query_count
    if tail_count == reference_count:
        return reference_tail
    return _guarded_padding(tokenizer, tail_count, filler, guard, label="reverse query tail")


def _assert_tail_compensates(
    spans: dict[str, dict[str, int]], reference_spans: dict[str, dict[str, int]], label: str
) -> None:
    # tail 起点与参考行相同，tail 与 query 的 token 总数与参考行相同
    if spans["tail"]["start"] != reference_spans["tail"]["start"] or (
        spans["tail"]["count"] + spans["query"]["count"]
        != reference_spans["tail"]["count"] + reference_spans["query"]["count"]
    ):
        raise ValueError(f"{label} 的 tail 没有抵消 query 长度变化")


def _build_row(
    reference_row: dict[str, Any],
    binding: dict[str, Any],
    tokenizer: Tokenizer,
    *,
    intro: str,
    tail: str,
) -> dict[str, Any]:
    target = _target_pair(binding)
    reference_components = reference_row["components"]
    use_chat_template = "chat_prefix" in reference_components
    query = _query_text(target["value"], use_chat_template)
    _assert_query_stable(
        tokenizer, query, int(target["value_token_id"]), reference_row["sample_id"]
    )
    replacements = {"intro": intro, "tail": tail, "query": query}
    reference_metadata = reference_row["metadata"]
    reference_spans = reference_metadata["spans"]
    # JSON 中的组件按名称排序，真实顺序由参考 span 的起点决定
    order = sorted(reference_components, key=lambda label: int(reference_spans[label]["start"]))
    components = {label: replacements.get(label, reference_components[label]) for label in order}
    prompt = "".join(components.values())
    component_ids = {label: token_ids(tokenizer, text) for label, text in components.items()}
    separate = [token for ids in component_ids.values() for token in ids]
    prompt_ids = token_ids(tokenizer, prompt)
    if prompt_ids != separate:
        raise ValueError(f"{reference_row['sample_id']} 的组件边界无法独立编码")
    if len(prompt_ids) != int(reference_metadata["prompt_token_count"]):
        raise ValueError(f"{reference_row['sample_id']} 的 prompt 长度与参考数据不一致")
    _assert_chat_template(tokenizer, components, prompt)
    starts: dict[str, int] = {}
    cursor = 0
    for label, ids in component_ids.items():
        starts[label] = cursor
        cursor += len(ids)
    spans = {
        label: {"start": starts[label], "count": len(component_ids[label])} for label in components
    }
    for label in components:
        if label not in REPLACED_COMPONENTS and spans[label] != reference_spans[label]:
            raise ValueError(f"{reference_row['sample_id']} 的组件 {label} 位置发生变化")
    if spans["intro"] != reference_spans["intro"]:
        raise ValueError(f"{reference_row['sample_id']} 的 intro 位置或长度发生变化")
    _assert_tail_compensates(spans, reference_spans, reference_row["sample_id"])
    target_pair_start = int(reference_metadata["target_pair_token_start"])
    target_value_start = int(reference_metadata["target_value_token_start"])
    if prompt_ids[target_pair_start] != int(target["key_token_id"]):
        raise ValueError(f"{reference_row['sample_id']} 的目标 key 位置不正确")
    if prompt_ids[target_value_start] != int(target["value_token_id"]):
        raise ValueError(f"{reference_row['sample_id']} 的目标 value 位置不正确")
    if token_ids(tokenizer, prompt + target["key"]) != [*prompt_ids, int(target["key_token_id"])]:
        raise ValueError(f"{reference_row['sample_id']} 的答案 key 无法追加")
    query_start = starts["query"]
    metadata = dict(reference_metadata)
    metadata.update(
        {
            "input_token_ids_sha256": token_ids_sha256(prompt_ids),
            "query_token_start": query_start,
            "query_token_count": len(component_ids["query"]),
            "target_to_query_distance": query_start - target_value_start,
            "query_direction": QUERY_DIRECTION,
            "spans": spans,
        }
    )
    output = dict(reference_row)
    output["answer"] = [target["key"]]
    output["components"] = components
    output["prompt"] = prompt
    output["metadata"] = metadata
    output["query_direction"] = QUERY_DIRECTION
    return output


def _query_protocol(
    tokenizer: Tokenizer, intro: str, tails: dict[str, str], query_count: int, reference: Path
) -> dict[str, Any]:
    return {
        "direction": QUERY_DIRECTION,
        "query_prefix": REVERSE_QUERY_PREFIX,
        "query_middle": REVERSE_QUERY_MIDDLE,
        "query_suffix": REVERSE_QUERY_SUFFIX,
        "intro_token_count": len(token_ids(tokenizer, intro)),
        "tail_token_counts": sorted({len(token_ids(tokenizer, tail)) for tail in tails.values()}),
        "query_token_count": query_count,
        "reference_dataset": str(reference.resolve()),
    }


def build_dataset(
    reference_root: str | Path,
    output_root: str | Path,
    tokenizer: Tokenizer,
    *,
    filler: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    reference = Path(reference_root)
    output = Path(output_root)
    if output.exists() and not overwrite:
        raise FileExistsError(f"输出目录已经存在：{output}")
    metadata = json.loads((reference / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("query_protocol", {}).get("direction") == QUERY_DIRECTION:
        raise ValueError("参考数据已经是反向 query 数据集")
    if hashlib.sha256(filler.encode("utf-8")).hexdigest() != metadata["filler_sha256"]:
        raise ValueError("filler 与参考数据使用的 filler 不一致")
    config = ExperimentConfig.from_dict(metadata["config"])
    layout = config.layout
    bindings = _load_bindings(reference)
    phases = [int(phase) for phase in metadata["matrix"]["phases"]]
    first_row = _read_jsonl(reference / "rows" / PAIR_FORM / f"phase_{phases[0]}.jsonl")[0]
    first_components = first_row["components"]
    use_chat_template = "chat_prefix" in first_components
    intro = build_reverse_intro(tokenizer, first_components["intro"], filler, layout)
    first_value = _target_pair(bindings[int(first_row["binding_id"])])["value"]
    query_count = len(token_ids(tokenizer, _query_text(first_value, use_chat_template)))
    # 参考数据的 tail 可能随行变化（随机目标数据集按相位偏移调整过 tail），按参考 tail 分别补偿
    tails: dict[str, str] = {}
    output.mkdir(parents=True, exist_ok=True)
    (output / "rows" / PAIR_FORM).mkdir(parents=True, exist_ok=True)
    total = 0
    for phase in phases:
        reference_rows = _read_jsonl(reference / "rows" / PAIR_FORM / f"phase_{phase}.jsonl")
        rows = []
        for reference_row in reference_rows:
            reference_tail = reference_row["components"]["tail"]
            if reference_tail not in tails:
                tails[reference_tail] = build_tail(
                    tokenizer,
                    reference_tail,
                    filler,
                    layout.tail_guard_token,
                    reference_query_count=len(
                        token_ids(tokenizer, reference_row["components"]["query"])
                    ),
                    query_count=query_count,
                )
            row = _build_row(
                reference_row,
                bindings[int(reference_row["binding_id"])],
                tokenizer,
                intro=intro,
                tail=tails[reference_tail],
            )
            if row["metadata"]["query_token_count"] != query_count:
                raise ValueError(f"{reference_row['sample_id']} 的 query token 数量与其他行不同")
            rows.append(row)
        _write_jsonl(output / "rows" / PAIR_FORM / f"phase_{phase}.jsonl", rows)
        total += len(rows)
    for name in ("bindings.jsonl", "sentence_schedule.jsonl", "padding_schedule.jsonl"):
        if (reference / name).is_file():
            (output / name).write_bytes((reference / name).read_bytes())
    output_metadata = dict(metadata)
    output_metadata["dataset_id"] = f"{metadata['dataset_id']}-reverse-query"
    output_metadata["query_protocol"] = _query_protocol(
        tokenizer, intro, tails, query_count, reference
    )
    output_metadata["reverse_query_rows"] = total
    (output / "metadata.json").write_text(
        json.dumps(output_metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_metadata


def validate_dataset(
    dataset_root: str | Path, reference_root: str | Path, tokenizer: Tokenizer
) -> dict[str, Any]:
    """逐行与参考数据比较：除 intro、tail、query 之外的组件逐字节相同，位置与长度全部一致。

    tail 与参考行逐条对应：起点相同，tail 与 query 的 token 总数相同。
    """

    dataset = Path(dataset_root)
    reference = Path(reference_root)
    metadata = json.loads((dataset / "metadata.json").read_text(encoding="utf-8"))
    reference_metadata = json.loads((reference / "metadata.json").read_text(encoding="utf-8"))
    protocol = metadata.get("query_protocol")
    if not isinstance(protocol, dict) or protocol.get("direction") != QUERY_DIRECTION:
        raise ValueError("metadata 中的反向 query 参数不正确")
    if (
        protocol.get("query_prefix"),
        protocol.get("query_middle"),
        protocol.get("query_suffix"),
    ) != (REVERSE_QUERY_PREFIX, REVERSE_QUERY_MIDDLE, REVERSE_QUERY_SUFFIX):
        raise ValueError("metadata 中的反向 query 模板不正确")
    if metadata["dataset_id"] != f"{reference_metadata['dataset_id']}-reverse-query":
        raise ValueError("dataset_id 与参考数据不对应")
    if (dataset / "bindings.jsonl").read_bytes() != (reference / "bindings.jsonl").read_bytes():
        raise ValueError("bindings.jsonl 与参考数据不一致")
    config = ExperimentConfig.from_dict(metadata["config"])
    bindings = _load_bindings(dataset)
    phases = [int(phase) for phase in metadata["matrix"]["phases"]]
    query_starts: set[int] = set()
    intro_texts: set[str] = set()
    checked = 0
    for phase in phases:
        rows = _read_jsonl(dataset / "rows" / PAIR_FORM / f"phase_{phase}.jsonl")
        reference_rows = _read_jsonl(reference / "rows" / PAIR_FORM / f"phase_{phase}.jsonl")
        if len(rows) != len(reference_rows):
            raise ValueError(f"phase {phase} 的行数与参考数据不一致")
        for row, reference_row in zip(rows, reference_rows, strict=True):
            label = row["sample_id"]
            for name in ("sample_id", "phase", "row_index", "binding_id", "pair_form"):
                if row[name] != reference_row[name]:
                    raise ValueError(f"{label} 的 {name} 与参考数据不一致")
            if row.get("query_direction") != QUERY_DIRECTION:
                raise ValueError(f"{label} 缺少反向 query 标记")
            binding = bindings[int(row["binding_id"])]
            target = _target_pair(binding)
            if row["answer"] != [target["key"]]:
                raise ValueError(f"{label} 的答案必须是目标 key")
            reference_components = reference_row["components"]
            reference_spans = reference_row["metadata"]["spans"]
            if set(row["components"]) != set(reference_components):
                raise ValueError(f"{label} 的组件集合与参考数据不一致")
            order = sorted(
                reference_components, key=lambda name: int(reference_spans[name]["start"])
            )
            components = {name: row["components"][name] for name in order}
            for name, text in components.items():
                if name not in REPLACED_COMPONENTS and text != reference_components[name]:
                    raise ValueError(f"{label} 的组件 {name} 与参考数据不一致")
            use_chat_template = "chat_prefix" in components
            if components["query"] != _query_text(target["value"], use_chat_template):
                raise ValueError(f"{label} 的 query 与目标 value 不一致")
            _assert_query_stable(
                tokenizer, components["query"], int(target["value_token_id"]), label
            )
            intro = components["intro"]
            intro_base = render_reverse_intro(config.layout)
            if not intro.startswith(intro_base) or not intro.endswith("\n"):
                raise ValueError(f"{label} 的 intro 文本不正确")
            prompt = "".join(components.values())
            if prompt != row["prompt"]:
                raise ValueError(f"{label} 的 prompt 与组件连接结果不一致")
            component_ids = {name: token_ids(tokenizer, text) for name, text in components.items()}
            prompt_ids = token_ids(tokenizer, prompt)
            separate = [token for ids in component_ids.values() for token in ids]
            if prompt_ids != separate:
                raise ValueError(f"{label} 的组件边界无法独立编码")
            if len(prompt_ids) != int(reference_row["metadata"]["prompt_token_count"]):
                raise ValueError(f"{label} 的 prompt 长度与参考数据不一致")
            _assert_chat_template(tokenizer, components, prompt)
            starts: dict[str, int] = {}
            cursor = 0
            for name, ids in component_ids.items():
                starts[name] = cursor
                cursor += len(ids)
            spans = {
                name: {"start": starts[name], "count": len(component_ids[name])}
                for name in components
            }
            for name in components:
                if name not in REPLACED_COMPONENTS and spans[name] != reference_spans[name]:
                    raise ValueError(f"{label} 的组件 {name} 位置与参考数据不一致")
            if spans["intro"] != reference_spans["intro"]:
                raise ValueError(f"{label} 的 intro 位置与参考数据不一致")
            _assert_tail_compensates(spans, reference_spans, label)
            expected_metadata = dict(reference_row["metadata"])
            expected_metadata.update(
                {
                    "input_token_ids_sha256": token_ids_sha256(prompt_ids),
                    "query_token_start": starts["query"],
                    "query_token_count": len(component_ids["query"]),
                    "target_to_query_distance": starts["query"]
                    - int(reference_row["metadata"]["target_value_token_start"]),
                    "query_direction": QUERY_DIRECTION,
                    "spans": spans,
                }
            )
            if row["metadata"] != expected_metadata:
                raise ValueError(f"{label} 的 metadata 与重新计算结果不一致")
            target_pair_start = int(row["metadata"]["target_pair_token_start"])
            if prompt_ids[target_pair_start] != int(target["key_token_id"]):
                raise ValueError(f"{label} 的目标 key 位置不正确")
            if prompt_ids[target_pair_start + 2] != int(target["value_token_id"]):
                raise ValueError(f"{label} 的目标 value 位置不正确")
            appended = [*prompt_ids, int(target["key_token_id"])]
            if token_ids(tokenizer, prompt + target["key"]) != appended:
                raise ValueError(f"{label} 的答案 key 无法追加")
            query_starts.add(starts["query"])
            intro_texts.add(components["intro"])
            checked += 1
    if len(query_starts) != 1 or len(intro_texts) != 1:
        raise ValueError("query 起点或 intro 随样本变化")
    expected_rows = {int(metadata["reverse_query_rows"]), int(metadata["matrix"]["total_rows"])}
    if expected_rows != {checked}:
        raise ValueError("检查的行数与 metadata 不一致")
    return {
        "status": "ok",
        "dataset_id": metadata["dataset_id"],
        "validated_rows": checked,
        "query_protocol": protocol,
        "query_start": query_starts.pop(),
    }


def _tokenizer_for(metadata_path: Path, tokenizer_path: Path) -> FastTokenizer:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    config = ExperimentConfig.from_dict(metadata["config"])
    return load_tokenizer(
        config.tokenizer.model_id, config.tokenizer.revision, local_path=tokenizer_path
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m periodic_weak_spots.niah_128k.reverse_query")
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="从参考数据集构造 value→key 的反向 query 数据集")
    build.add_argument("--reference-dataset", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--tokenizer-path", type=Path, required=True)
    build.add_argument(
        "--filler-file", type=Path, required=True, help="与参考数据相同的 filler 文本"
    )
    build.add_argument("--overwrite", action="store_true")
    validate = commands.add_parser("validate", help="逐行与参考数据集比较")
    validate.add_argument("--dataset", type=Path, required=True)
    validate.add_argument("--reference-dataset", type=Path, required=True)
    validate.add_argument("--tokenizer-path", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        tokenizer = _tokenizer_for(args.reference_dataset / "metadata.json", args.tokenizer_path)
        metadata = build_dataset(
            args.reference_dataset,
            args.output,
            tokenizer,
            filler=args.filler_file.read_bytes().decode("utf-8"),
            overwrite=args.overwrite,
        )
        print(json.dumps(metadata["query_protocol"], ensure_ascii=False, indent=2, sort_keys=True))
        return
    tokenizer = _tokenizer_for(args.dataset / "metadata.json", args.tokenizer_path)
    report = validate_dataset(args.dataset, args.reference_dataset, tokenizer)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
