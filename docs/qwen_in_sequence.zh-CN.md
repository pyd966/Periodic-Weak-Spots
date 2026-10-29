# Qwen3.8-Flash-Next In-sequence Padding

In-sequence 数据集使用当前 Qwen3.8 Prefix Padding 数据集作为参考，保持相同的 binding、phase、row 映射、目标位置分布和 query 起点。固定前后 guard 各使用8个 `｜`，record 之间额外插入32个 `｜`，每个间隔最多插入4个。Qwen3.8 Tokenizer 中 `｜` 的 token ID 为158819。

对于参考数据中的 `reference_prefill_token_count=L`，目标前的 gap 总长度为 `L-8`，目标后的 gap 总长度为 `32-(L-8)`。gap 向量使用确定性的 bounded composition 生成，向量包含15999个间隔，对应16000条 record 之间的位置。

构造入口会写入独立目录，不读取自然语言句子来源，也不会修改当前 Prefix Padding 数据集：

```bash
source /root/autodl-tmp/venvs/periodic-weak-spots-vllm/bin/activate
PYTHONPATH=/root/Periodic-Weak-Spots python -m periodic_weak_spots.niah_128k.in_sequence build \
  --reference-dataset /root/Periodic-Weak-Spots/.research/niah_qwen_128k_512 \
  --output /root/autodl-tmp/periodic_weak_spots/niah_qwen_128k_in_sequence \
  --tokenizer-path /root/Periodic-Weak-Spots/resources/qwen3_8_flash_next
```

独立验证入口为：

```bash
PYTHONPATH=/root/Periodic-Weak-Spots python -m periodic_weak_spots.niah_128k.in_sequence validate \
  --dataset /root/autodl-tmp/periodic_weak_spots/niah_qwen_128k_in_sequence \
  --tokenizer-path /root/Periodic-Weak-Spots/resources/qwen3_8_flash_next
```

验证会重新编码每条完整 prompt，检查 guard、全部 gap、目标 key/value 位置、128000 token 长度、127958 query 起点、Qwen non-thinking chat template、token ID hash 和 phase 长度直方图。

评分时继续使用现有 vLLM 入口，只替换 dataset、progress 和 results 路径：

```bash
PYTHONPATH=/root/Periodic-Weak-Spots python -m periodic_weak_spots.niah_128k score \
  --dataset /root/autodl-tmp/periodic_weak_spots/niah_qwen_128k_in_sequence \
  --model /root/autodl-tmp/models/Qwen3.8-Flash-Next-FP8 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --max-model-len 131072 \
  --max-tokens 8 \
  --batch-size 16 \
  --progress results/qwen_niah_128k_fp8_tp4_in_sequence.progress.jsonl \
  --results results/qwen_niah_128k_fp8_tp4_in_sequence.json
```
