# Qwen3.8-Flash-Next 反向 query（给出 value 问 key）

反向 query 数据集以已经构造好的 Qwen3.8 数据集作为参考，Prefix Padding 与 In-sequence Padding 的数据集都可以作为参考。参考数据的 binding、record 顺序、每条 record 的位置、目标 record 的位置、前后填充、chat template 全部保持不变，只替换三个组件：

| 组件 | 做法 |
| --- | --- |
| `intro` | 指令改为 "I will give you a value afterwards and ask for its key."，其余句子与格式示例不变；用与参考数据相同的 filler 补齐到参考 intro 的 token 数量，因此 intro 之后所有组件的起点不变 |
| `query` | `Now answer the question using the key-value pairs above. What is the key for value {value}? Answer with only the key. The key for value {value} is`，三段模板的 token 数量与正向模板相同，query 共 33 token，起点仍为 127,958 |
| `tail` | query 的 token 数量与参考相同时逐字节沿用参考 tail；若模板改动造成 query 长度变化，tail 用相同 filler 重新构造以抵消差值 |

答案为目标 key。每一行记录 `query_direction = "value_to_key"`，评分入口按这个字段选择 query 模板与 gold，正向数据集没有这个字段，按原来的方式评分。

构造时逐行检查：组件边界独立编码、prompt 为 128,000 token、未替换组件的 span 与参考逐项相同、intro 的 span 与参考相同、目标 key 与 value 的位置与参考相同、value 在 query 中两处都是单个 token、目标 key 可以追加到 prompt 之后、完整文本与 Qwen non-thinking chat template 一致。`sample_id` 与参考数据相同，便于逐条配对比较。

构造入口写入独立目录，不修改参考数据集：

```bash
PYTHONPATH=/root/Periodic-Weak-Spots python -m periodic_weak_spots.niah_128k.reverse_query build \
  --reference-dataset /root/Periodic-Weak-Spots/.research/niah_qwen_128k_512 \
  --output /root/autodl-tmp/periodic_weak_spots/niah_qwen_128k_512_reverse \
  --tokenizer-path /root/Periodic-Weak-Spots/resources/qwen3_8_flash_next \
  --filler-file /root/Periodic-Weak-Spots/resources/niah_128k/paul_graham/filler.txt
```

In-sequence 参考数据只需替换 `--reference-dataset` 与 `--output`。filler 必须与参考数据构造时使用的文件相同，构造前会比较 SHA256。

独立验证需要同时给出参考数据集，逐行比较除 `intro`、`tail`、`query` 之外的组件是否逐字节相同，并重新编码检查全部位置、hash、答案 token、chat template 以及 query 起点、intro、tail 在全部样本中是否唯一：

```bash
PYTHONPATH=/root/Periodic-Weak-Spots python -m periodic_weak_spots.niah_128k.reverse_query validate \
  --dataset /root/autodl-tmp/periodic_weak_spots/niah_qwen_128k_512_reverse \
  --reference-dataset /root/Periodic-Weak-Spots/.research/niah_qwen_128k_512 \
  --tokenizer-path /root/Periodic-Weak-Spots/resources/qwen3_8_flash_next
```

评分继续使用现有 vLLM 入口，只替换 dataset、progress 和 results 路径：

```bash
PYTHONPATH=/root/Periodic-Weak-Spots python -m periodic_weak_spots.niah_128k score \
  --dataset /root/autodl-tmp/periodic_weak_spots/niah_qwen_128k_512_reverse \
  --model /root/autodl-tmp/models/Qwen3.8-Flash-Next-FP8 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --max-model-len 131072 \
  --max-tokens 8 \
  --batch-size 16 \
  --progress results/qwen_niah_128k_fp8_tp4_reverse.progress.jsonl \
  --results results/qwen_niah_128k_fp8_tp4_reverse.json
```

结果文件的 `records[*].input.target_key`、`target_value` 的位置与正向数据集逐条相同，`gold_answer` 为 key。
