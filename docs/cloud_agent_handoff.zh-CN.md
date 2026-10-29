# 云服务器交接说明

这份说明写给在云服务器上接手本仓库的 agent。仓库根目录的 `AGENTS.md` 对云服务器上的工作同样有效，开始之前先完整阅读它。

## 1. 项目是什么

论文 Zhu et al. 2026, *Periodic Weak Spots: Phase Sensitivity from Chunked KV-Cache Compression*（arXiv 2609.36322，PDF 在 `paper/`）发现：使用分块 KV-cache 压缩的模型，长上下文检索准确率随目标 token 的相位 `t mod S` 周期变化。本仓库是论文官方代码的一个 fork，把论文的 128K needle-in-a-haystack（NIAH）实验搬到 `Qwen/Qwen3.8-Flash-Next` 上。这个模型没有 KV 压缩，只有 QSA indexer 把每 4 个 token 平均成一个 block 再选 top-512 个 block，因此它能单独检验"block 选择"这一个因素。

已经完成的两组评分（`results/qwen_niah_128k_fp8_tp4_full.json` 与 `results/qwen_niah_128k_fp8_tp4_in_sequence.json`）得到：周期为 4，弱相位只有 `t_K mod 4 = 3`，相位 3 的准确率约 95.7%，其余相位约 99.2%，两种 padding 方式一致。分析与猜测见 `reports/qwen_niah_20261028_discussion.zh-CN.md`。当前假设是：目标 key 处于 block 末尾时，它所在 block 的平均表示被前一条 record 的三个 token 稀释，indexer 偶尔选不中它。

## 2. 压缩包里有什么

压缩包 `Periodic-Weak-Spots-cloud-20261029.tar.zst` 解压后是完整的仓库目录 `Periodic-Weak-Spots/`，包含 `.git`（分支 `main`，远端 `git@github.com:pyd966/Periodic-Weak-Spots.git`）、代码、配置、文档、论文、已有结果、Tokenizer 资源、分析脚本，以及八个已经构造并验证过的数据集。数据集目录在 `.gitignore` 范围内，不会进入 git。

解压命令（`zstd` 与 `tar` 都是 Ubuntu 标准软件包）：

```bash
zstd -dc Periodic-Weak-Spots-cloud-20261029.tar.zst | tar -x -C /root/autodl-tmp
cd /root/autodl-tmp/Periodic-Weak-Spots
sha256sum -c .research/cloud_bundle_sha256.txt --quiet && echo "bundle ok"
```

`.research/cloud_bundle_sha256.txt` 记录了八个数据集全部文件与 `results/` 的 SHA256。校验通过后不需要再运行各数据集的 validate 入口。

### 2.1 八个数据集

每个数据集 4,096 条 prompt，每条恰好 128,000 token，八个 phase 各 512 条。三个二元因素组合成八个数据集：

| 编号 | 目标 record | padding | query 方向 | 目录（相对仓库根目录） | 已评分 |
| --- | --- | --- | --- | --- | --- |
| A | 固定第 249 条 | Prefix | key → value | `.research/niah_qwen_128k_512` | 是 |
| B | 固定第 249 条 | In-sequence | key → value | `periodic_weak_spots/niah_qwen_128k_in_sequence` | 是 |
| C | 固定第 249 条 | Prefix | value → key | `.research/niah_qwen_128k_512_reverse` | 否 |
| D | 固定第 249 条 | In-sequence | value → key | `.research/niah_qwen_128k_in_sequence_reverse` | 否 |
| E | 每个 binding 随机一条 | Prefix | key → value | `.research/niah_qwen_128k_512_random_target` | 否 |
| F | 每个 binding 随机一条 | Prefix | value → key | `.research/niah_qwen_128k_512_random_target_reverse` | 否 |
| G | 每个 binding 随机一条 | In-sequence | key → value | `.research/niah_qwen_128k_random_target_in_sequence` | 否 |
| H | 每个 binding 随机一条 | In-sequence | value → key | `.research/niah_qwen_128k_random_target_in_sequence_reverse` | 否 |

八个数据集共用同一批 512 个 binding、同一套 phase 与 row 映射、同一个 query 起点 127,958 和同一批 `sample_id`，任意两组都可以逐条配对。A、B、C、D 的目标 key 位置逐条相同（均值 64,000）；E、F、G、H 的目标 key 位置逐条相同（范围 63,097 到 126,875）。各数据集的构造方式见 `docs/qwen_niah_adaptation.zh-CN.md`、`docs/qwen_in_sequence.zh-CN.md`、`docs/qwen_reverse_query.zh-CN.md`、`docs/qwen_random_target.zh-CN.md`。

反向 query 数据集的每一行带 `query_direction = "value_to_key"`，评分入口按这个字段选择 gold，不需要额外参数。

## 3. 环境

上一次评分使用的环境记录在 `periodic_weak_spots_bundle_metadata/`（`pip_freeze.txt`、`nvidia_smi.txt`、`commands.txt`）：

| 项目 | 设置 |
| --- | --- |
| GPU | 4 张 NVIDIA RTX PRO 6000 Blackwell，driver 580.76，CUDA 13.0 |
| Python | 3.12 |
| vLLM / torch / transformers | 0.32.0 / 2.13.0 / 5.19.0 |
| 模型 | `Qwen/Qwen3.8-Flash-Next` 官方 FP8 checkpoint，放在 `/root/autodl-tmp/models/Qwen3.8-Flash-Next-FP8` |
| 推理参数 | `--tensor-parallel-size 4 --enable-expert-parallel --max-model-len 131072 --max-tokens 8 --batch-size 16` |

评分入口在模型运行前检查模型目录中 `tokenizer.json`、`tokenizer_config.json`、`chat_template.jinja` 的 SHA256，与 `resources/qwen3_8_flash_next/` 中的固定 revision `de4b8e4d43b917e7706784d8bb445c9af86a3540` 不一致时直接终止。

安装：

```bash
cd /root/autodl-tmp/Periodic-Weak-Spots
uv venv --python 3.12 /root/autodl-tmp/venvs/periodic-weak-spots-vllm
source /root/autodl-tmp/venvs/periodic-weak-spots-vllm/bin/activate
uv pip install -e . "vllm>=0.32"
export PYTHONPATH=/root/autodl-tmp/Periodic-Weak-Spots
```

vLLM 的启动参数按官方 recipe 调整：<https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next>。

安装后用一个已有结果做冒烟检查：

```bash
python -m pytest -q .research/test_scoring_records.py .research/test_reverse_query.py .research/test_random_target.py -p no:cacheprovider
```

## 4. 要做的事：按顺序评分六个数据集

每次评分约 5 到 10 小时（A 用了 9.5 小时，B 用了 5.2 小时）。评分命令只替换 `--dataset`、`--progress`、`--results` 三个参数，其余与 A、B 完全相同：

```bash
python -m periodic_weak_spots.niah_128k score \
  --dataset <数据集目录> \
  --model /root/autodl-tmp/models/Qwen3.8-Flash-Next-FP8 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --max-model-len 131072 \
  --max-tokens 8 \
  --batch-size 16 \
  --progress results/<名称>.progress.jsonl \
  --results results/<名称>.json
```

按下面的顺序执行，结果文件名固定：

| 顺序 | 数据集 | `--dataset` | `<名称>` | 这一组回答什么 |
| --- | --- | --- | --- | --- |
| 1 | C | `.research/niah_qwen_128k_512_reverse` | `qwen_niah_128k_fp8_tp4_reverse` | 把要匹配的 token 从 key 换成 value 之后，弱相位是否随之移动。value 的相位是 `t_K + 2`；若弱相位变成 `t_K mod 4 = 1`，说明效应跟随"被匹配 token 处于 block 末尾"；若弱相位出现在 `t_K mod 4 ∈ {2, 3}`，说明效应跟随"答案 token 所在 block 位于被匹配 block 之前"。与 A 逐条配对 |
| 2 | E | `.research/niah_qwen_128k_512_random_target` | `qwen_niah_128k_fp8_tp4_random_target` | 目标位置在整段 record body 中均匀分布时，相位 3 的下降是否仍然存在，以及下降幅度是否随目标到 query 的距离变化。与 A 共用 binding，目标 record 不同 |
| 3 | F | `.research/niah_qwen_128k_512_random_target_reverse` | `qwen_niah_128k_fp8_tp4_random_target_reverse` | C 的结论在随机位置上是否成立。与 E 逐条配对 |
| 4 | D | `.research/niah_qwen_128k_in_sequence_reverse` | `qwen_niah_128k_fp8_tp4_in_sequence_reverse` | 反向 query 下 distractor 相位被打散后结果是否不变。与 B、C 逐条配对 |
| 5 | G | `.research/niah_qwen_128k_random_target_in_sequence` | `qwen_niah_128k_fp8_tp4_random_target_in_sequence` | 随机位置下的 In-sequence 对照。与 E 逐条配对 |
| 6 | H | `.research/niah_qwen_128k_random_target_in_sequence_reverse` | `qwen_niah_128k_fp8_tp4_random_target_in_sequence_reverse` | 随机位置、反向 query 的 In-sequence 对照。与 F、G 逐条配对 |

C 与 E 放在最前面，因为它们各自检验一个独立的问题，并且都只依赖已经评分的 A。In-sequence 的三组放在最后，因为 B 与 A 的差异没有统计意义（McNemar `p ≈ 0.09`），它们主要用于确认结论不依赖 distractor 的相位。

六组完成后，若 GPU 仍可用，再在 A 上运行 oracle indexer（先对每个 token 用 indexer 的 q、k 打分，再按 block 求和；见 `docs/qwen_oracle_indexer.zh-CN.md`），结果写到 `results/qwen_niah_128k_fp8_tp4_oracle.json`。相位 3 在 oracle 下恢复，说明下降来自"先平均再打分"这一步。

进度页面：

```bash
python -m periodic_weak_spots.niah_128k.monitor --progress results/<名称>.progress.jsonl --host 0.0.0.0 --port 8765
```

## 5. 每组评分结束后的检查

1. `results/<名称>.json` 可以用标准 JSON 解析器读取，`schema_version` 为 2，`summary.rows` 为 4096，`records` 共 4096 条且 `sample_id` 唯一，八个 phase 各 512 条。
2. 反向 query 数据集（C、D、F、H）的 `records[*].gold_answer` 应为 key；正向数据集应为 value。
3. 把 `summary.answer_prefix_accuracy_by_residue`、`correct_count_by_residue`、`gap` 记入提交信息。
4. 提交到 `main` 分支并推送：结果 JSON、progress JSONL 与运行日志一起提交，提交信息使用 Conventional Commits（例如 `feat(results): score reverse query dataset on Qwen3.8-Flash-Next FP8`），结尾加 `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`。数据集目录不进入 git。

```bash
git add results/<名称>.json results/<名称>.progress.jsonl
git commit
git push origin main
```

## 6. 约束

- 不修改任何数据集文件、Tokenizer、chat template 或 `configs/`。评分入口会在实际 token ID 与构造记录不一致时终止，终止后保留完整日志，记录位置与错误信息，不要改动 prompt 后继续使用同一个结果文件。
- 中间文件、日志、临时脚本放在 `.research/` 下，这个目录在 `.gitignore` 范围内。
- 推理中断后重新运行整组评分，不要拼接两个不完整的结果文件。
- 代码改动遵守 `AGENTS.md`，先运行 `.research/test_*.py` 再提交。
