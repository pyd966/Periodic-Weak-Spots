# Qwen3.8-Flash-Next 随机目标位置数据集

随机目标数据集以 Qwen3.8 Prefix Padding 数据集作为参考。参考数据的 512 个 binding、每条 record 的内容与顺序、自然语言前后句、chat template、query 起点全部保持不变；每个 binding 用固定 seed 均匀抽取一条 record 作为目标，同一个 binding 在八个 phase 下使用同一个目标，因此八个 phase 之间仍然按 binding 逐条配对。

## 相位控制

目标 record 相对 record body 起点的偏移为 `4t`，`t` 为目标下标。偏移在 mod 8 下为 0（`t` 为偶数）或 4（`t` 为奇数）。偏移为 4 时，`body_prefix` 缩短 4 个 token，`tail` 延长 4 个 token；record body、自然语言后句整体向前移动 4 个 token，prompt 总长度、query 起点、query 之后的组件都不变。目标 key 的相位因此仍由自然语言前句长度决定，`phase_control_prefix_count` 与参考数据相同。

| 组件 | `t` 为偶数 | `t` 为奇数 |
| --- | --- | --- |
| `chat_prefix`、`intro`、`natural_prefill` | 与参考相同 | 与参考相同 |
| `body_prefix` | 与参考相同（62,880 token） | 同一 filler 重新构造，62,876 token |
| `pair_body`、`natural_postfill` | 与参考相同 | 文本相同，起点前移 4 |
| `tail` | 与参考相同（925 token） | 同一 filler 重新构造，929 token |
| `query_separator`、`query`、`chat_suffix` | 位置与参考相同，query 文本改为目标 key | 同上 |

目标 key 的绝对位置为 `pair_body 起点 + 4t`，在整段 record body 中近似均匀分布，与 query 的距离随之变化。

## 目标抽取

`random.Random(f"{seed}:random-target:{pair_count}")` 按 `binding_id` 升序为每个 binding 抽取 `randrange(pair_count)`。被抽到的 record 必须同时满足三个条件：下标在 `[8, 15991]` 之内（In-sequence Padding 把至多 32 个 `｜` 放进目标一侧的间隔，每个间隔至多 4 个，目标两侧因此都至少需要 8 条 record）；key 在正向 query 模板中保持单个 token；value 在反向 query 模板中保持单个 token（两种模板的原始形式与 `rstrip` 形式都检查）。不满足时重新抽取，被拒绝的次数记录在 metadata 中。这样同一组目标可以直接用于反向 query 与 In-sequence Padding 数据集。默认 seed 为 20261029。

输出的 `bindings.jsonl` 保留参考数据的全部 record，只更新 `target_index`、`target_key`、`target_value`。每一行的 `metadata` 记录 `target_index`、`target_pair_token_start`、`target_value_token_start`、`target_phase`、`target_protocol = "random_index"`、`body_prefix_shift_tokens` 与全部组件 span。数据集 `metadata.json` 新增 `target_protocol`，包含 seed、两侧保留的 record 数量、被拒绝的抽样次数、偏移 4 的 binding 数量、目标下标与目标位置的最小值、最大值、平均值；`design.target_position_moments` 改为实际的目标位置统计。`config` 与参考数据相同，评分入口不需要改动。

## 构造与验证

```bash
PYTHONPATH=/root/Periodic-Weak-Spots python -m periodic_weak_spots.niah_128k.random_target build \
  --reference-dataset /root/Periodic-Weak-Spots/.research/niah_qwen_128k_512 \
  --output /root/autodl-tmp/periodic_weak_spots/niah_qwen_128k_512_random_target \
  --tokenizer-path /root/Periodic-Weak-Spots/resources/qwen3_8_flash_next \
  --filler-file /root/Periodic-Weak-Spots/resources/niah_128k/paul_graham/filler.txt
```

构造前会比较 filler 的 SHA256，并用该 filler 重新构造参考数据的 `body_prefix` 与 `tail`，要求逐字节相同。每一行检查：组件边界独立编码、prompt 为 128,000 token、完整文本与 Qwen non-thinking chat template 一致、每个组件的起点与长度符合上表、目标 key 与 value 的 token 位于计算出的位置、`target_pair_token_start mod 8` 等于行的 phase、答案 value 可以追加到 prompt 之后。

验证用参考数据、filler 与 metadata 中的 seed 重新抽取目标并逐行重新构造，要求 `bindings.jsonl`、`sentence_schedule.jsonl` 与每一行都与数据集完全相同：

```bash
PYTHONPATH=/root/Periodic-Weak-Spots python -m periodic_weak_spots.niah_128k.random_target validate \
  --dataset /root/autodl-tmp/periodic_weak_spots/niah_qwen_128k_512_random_target \
  --reference-dataset /root/Periodic-Weak-Spots/.research/niah_qwen_128k_512 \
  --tokenizer-path /root/Periodic-Weak-Spots/resources/qwen3_8_flash_next \
  --filler-file /root/Periodic-Weak-Spots/resources/niah_128k/paul_graham/filler.txt
```

## 反向 query 版本

随机目标数据集可以直接作为 [反向 query](qwen_reverse_query.zh-CN.md) 的参考数据集，目标、位置、padding 全部沿用，只替换 intro、query 与 tail 的补偿部分：

```bash
PYTHONPATH=/root/Periodic-Weak-Spots python -m periodic_weak_spots.niah_128k.reverse_query build \
  --reference-dataset /root/autodl-tmp/periodic_weak_spots/niah_qwen_128k_512_random_target \
  --output /root/autodl-tmp/periodic_weak_spots/niah_qwen_128k_512_random_target_reverse \
  --tokenizer-path /root/Periodic-Weak-Spots/resources/qwen3_8_flash_next \
  --filler-file /root/Periodic-Weak-Spots/resources/niah_128k/paul_graham/filler.txt
```

## In-sequence Padding 版本

随机目标数据集也可以作为 [In-sequence Padding](qwen_in_sequence.zh-CN.md) 的参考数据集。In-sequence 构造器按行读取目标下标、`body_prefix` 与 `tail`，目标之前的间隔共放入 `L − 8` 个 `｜`，目标 key 位置与随机目标数据集逐条相同；行的 `target_protocol`、`body_prefix_shift_tokens` 标记随行保留。再对该数据集运行反向 query 构造器即得到 In-sequence 随机目标的反向版本：

```bash
PYTHONPATH=/root/Periodic-Weak-Spots python -m periodic_weak_spots.niah_128k.in_sequence build \
  --reference-dataset /root/autodl-tmp/periodic_weak_spots/niah_qwen_128k_512_random_target \
  --output /root/autodl-tmp/periodic_weak_spots/niah_qwen_128k_random_target_in_sequence \
  --tokenizer-path /root/Periodic-Weak-Spots/resources/qwen3_8_flash_next

PYTHONPATH=/root/Periodic-Weak-Spots python -m periodic_weak_spots.niah_128k.reverse_query build \
  --reference-dataset /root/autodl-tmp/periodic_weak_spots/niah_qwen_128k_random_target_in_sequence \
  --output /root/autodl-tmp/periodic_weak_spots/niah_qwen_128k_random_target_in_sequence_reverse \
  --tokenizer-path /root/Periodic-Weak-Spots/resources/qwen3_8_flash_next \
  --filler-file /root/Periodic-Weak-Spots/resources/niah_128k/paul_graham/filler.txt
```

## 评分

评分继续使用现有 vLLM 入口，只替换 dataset、progress 和 results 路径：

```bash
PYTHONPATH=/root/Periodic-Weak-Spots python -m periodic_weak_spots.niah_128k score \
  --dataset /root/autodl-tmp/periodic_weak_spots/niah_qwen_128k_512_random_target \
  --model /root/autodl-tmp/models/Qwen3.8-Flash-Next-FP8 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --max-model-len 131072 \
  --max-tokens 8 \
  --batch-size 16 \
  --progress results/qwen_niah_128k_fp8_tp4_random_target.progress.jsonl \
  --results results/qwen_niah_128k_fp8_tp4_random_target.json
```

## 全部 Qwen3.8 数据集之间的关系

三个二元因素（padding 方式、query 方向、目标位置）组合成八个数据集：

| 目标 record | padding | query 方向 | 参考数据集 |
| --- | --- | --- | --- |
| 固定第 249 条（`target_index = 248`） | Prefix | key → value | 由 build 直接生成 |
| 固定第 249 条 | In-sequence | key → value | 固定 / Prefix / 正向 |
| 固定第 249 条 | Prefix | value → key | 固定 / Prefix / 正向 |
| 固定第 249 条 | In-sequence | value → key | 固定 / In-sequence / 正向 |
| 每个 binding 随机一条 | Prefix（奇数下标时 body_prefix 与 tail 各调整 4 token） | key → value | 固定 / Prefix / 正向 |
| 每个 binding 随机一条 | In-sequence | key → value | 随机 / Prefix / 正向 |
| 每个 binding 随机一条 | Prefix | value → key | 随机 / Prefix / 正向 |
| 每个 binding 随机一条 | In-sequence | value → key | 随机 / In-sequence / 正向 |

八个数据集共用同一批 512 个 binding（record 内容与顺序相同）、同一套 phase 与 row 映射、同一批自然语言句子长度、同一个 query 起点 127,958 和 4,096 个 `sample_id`。固定目标的四个数据集目标 key 位置逐条相同；随机目标的四个数据集目标 key 位置逐条相同，与固定目标的不同。同一目标位置下，Prefix 与 In-sequence 的区别只在 padding 的形式，正向与反向的区别只在 intro 的一句话与 query 模板。
