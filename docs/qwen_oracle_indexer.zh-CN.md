# Qwen3.8-Flash-Next oracle indexer

`periodic_weak_spots.niah_128k.oracle_vllm` 提供一个独立的 vLLM model class override。运行它时，Qwen 原有的 QSA attention、KV cache、causal attention 和后续模型计算继续使用 vLLM 实现；QSA 的 block 选择由 `OracleQSAIndexer` 重写。

对于每个最终 query row，代码执行以下计算：

1. 使用 QSA indexer 的 `q_layernorm`、`k_layernorm` 和 RoPE 得到 query 与每个历史 key。
2. 对每个 indexer head 计算所有 token 的 `q · k`，将负分数截断为零，并按照 Qwen 原 indexer 的 head 聚合规则求和。
3. 将连续四个 token 的分数相加，得到 block score。
4. 按 block score 选择 `indexer_budget / indexer_compress_ratio` 个 block。
5. 按 Qwen 原有的 block 展开规则加入选中 token 与 causal tail，然后继续原有 attention 流程。

当前实现对每个 forward 中满足 `logical_position == sequence_length - 1` 的 row 使用 oracle 选择。这覆盖 NIAH 生成答案时的最终 prompt query 和之后的 decode query；prefill 中间 row 仍然使用原有 indexer，以避免为每一个中间 query 进行完整的历史 key 矩阵计算。历史 QSA indexer key 使用 CPU `bfloat16` 保存，最终 query 选择时传回 GPU 计算，因此显存额外占用低，运行时间会明显增加。

## 运行命令

当前评测结束后，可以使用单独的结果文件运行：

```bash
source /root/autodl-tmp/venvs/periodic-weak-spots-vllm/bin/activate
PYTHONPATH=/root/Periodic-Weak-Spots python -m periodic_weak_spots.niah_128k.oracle_score \
  --dataset .research/niah_qwen_128k_512 \
  --model /root/autodl-tmp/models/Qwen3.8-Flash-Next-FP8 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --max-model-len 131072 \
  --max-tokens 8 \
  --batch-size 16 \
  --progress results/qwen_niah_128k_fp8_tp4_oracle.progress.jsonl \
  --results results/qwen_niah_128k_fp8_tp4_oracle.json
```

`oracle_score` 会写入每条完整结果记录、top10 next-token 概率、阶段统计和 oracle 配置。`--progress` 文件可以直接交给现有 monitor 页面读取：

```bash
PYTHONPATH=/root/Periodic-Weak-Spots python -m periodic_weak_spots.niah_128k.monitor \
  --progress results/qwen_niah_128k_fp8_tp4_oracle.progress.jsonl \
  --host 0.0.0.0 \
  --port 8766
```

代码通过 `model_class_overrides` 注册 override，没有修改 vLLM 安装目录，也没有修改当前普通 indexer 的评分入口。`enforce_eager=True` 与 `enable_prefix_caching=False` 是这个 baseline 的固定设置，用于支持按请求保存历史 key。
