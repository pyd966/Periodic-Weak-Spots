# 云端执行 Qwen3.8-Flash-Next NIAH

这份说明用于在另一台云服务器上执行已经生成好的 128K NIAH。执行范围固定为 `Qwen/Qwen3.8-Flash-Next`、8 个 phase、每个 phase 512 条 prompt，共 4096 条。

## 需要带到云服务器的文件

云服务器需要同时拥有仓库代码、prompt 数据和模型目录。

当前生成数据位于：

```text
.research/niah_qwen_128k_512/
```

这个目录约 5.2 GB，整个 `.research/` 已经加入 `.gitignore`，Git 工作树中不会自动包含它。至少需要复制以下目录：

```text
.research/niah_qwen_128k_512/
resources/qwen3_8_flash_next/
```

`resources/qwen3_8_flash_next/` 用于检查固定 revision 的 Tokenizer。`resources/` 中的句子文件和 Paul Graham essays 只在云服务器重新生成 prompt 时需要；已经复制完整 prompt 后，评分过程只读取 prompt 数据和模型目录。

复制完成后，应当看到：

```text
.research/niah_qwen_128k_512/metadata.json
.research/niah_qwen_128k_512/bindings.jsonl
.research/niah_qwen_128k_512/sentence_schedule.jsonl
.research/niah_qwen_128k_512/rows/key_value/phase_0.jsonl
...
.research/niah_qwen_128k_512/rows/key_value/phase_7.jsonl
```

每个 `phase_*.jsonl` 文件应当包含 512 行。每一行的 `prompt` 字段已经包含完整的 Qwen user、assistant 和 non-thinking 标记，可以直接作为模型输入。

## 下载模型权重

在云服务器上使用固定 revision 下载默认 BF16 模型：

```bash
hf download Qwen/Qwen3.8-Flash-Next \
  --revision de4b8e4d43b917e7706784d8bb445c9af86a3540 \
  --local-dir /models/Qwen3.8-Flash-Next
```

模型目录需要包含 `config.json`、`tokenizer.json`、`tokenizer_config.json` 和 `chat_template.jinja`。评分入口会检查三个 Tokenizer 文件的 SHA256；文件版本不一致时会立即终止。

如果使用官方 FP8 checkpoint，需要把 `--model` 指向对应的本地目录，并按照云服务器 GPU 的要求启用 expert parallel。官方 recipe 说明了 FP8、BF16、GPU 数量、显存和 vLLM 参数：[Qwen3.8-Flash-Next vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next)。当前仓库的评分入口提供 `--enable-expert-parallel`；其他硬件专用参数需要由云服务器上的 agent 根据 recipe 配置。

## 安装推理环境

在仓库根目录创建独立的 vLLM 环境：

```bash
uv venv --python 3.12 .venv-vllm
uv pip install --python .venv-vllm/bin/python -e . "vllm>=0.30"
```

官方 recipe 标注稳定版 vLLM `0.30.0+` 可以运行，nightly 版本经过专门验证。Qwen 的 128K 输入需要足够的 GPU 显存和主机内存；启动参数应当遵循官方 recipe。[官方 recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next)

## 运行完整输入检查

进入仓库根目录，使用云服务器上的 Tokenizer 目录检查全部 prompt：

```bash
source .venv-vllm/bin/activate

python -m periodic_weak_spots.niah_128k.check_qwen \
  --dataset .research/niah_qwen_128k_512 \
  --tokenizer-path /models/Qwen3.8-Flash-Next \
  --report reports/qwen_niah_validation_cloud.json
```

检查程序会重新编码全部 4096 条 prompt，并检查：

- 每条输入是否为 128000 个 token。
- 官方 non-thinking chat template 是否逐条一致。
- 目标 key、目标 value、答案和 query 的 token 位置。
- 目标位置的八个 phase、均值 64000 和方差 28。
- 每个 phase 是否正好包含 512 条 prompt。
- HF 与 GigaToken 的完整 token ID 是否一致。

检查成功后，报告中的 `validated_rows` 应为 `4096`，每个 phase 的 `rows` 应为 `512`，`leading_special_ids` 应为空列表。

## 执行评分

默认 BF16 模型可以使用以下命令：

```bash
python -m periodic_weak_spots.niah_128k score \
  --dataset .research/niah_qwen_128k_512 \
  --model /models/Qwen3.8-Flash-Next \
  --tensor-parallel-size 4 \
  --max-model-len 131072 \
  --max-tokens 8 \
  --results results/qwen_niah_128k.json
```

如果 GPU 配置要求 expert parallel，增加参数：

```bash
  --enable-expert-parallel
```

`score` 使用 vLLM 的离线 `LLM.generate`，不启动 OpenAI 兼容服务。程序在模型运行前使用模型目录中的 Tokenizer 编码 prompt，再通过 vLLM `TokensPrompt` 传入 `prompt_token_ids`。vLLM 的 token 输入定义见[官方输入接口说明](https://docs.vllm.ai/en/stable/api/vllm/inputs/)。

评分参数固定为：

- `generation_config="vllm"`。
- `temperature=0`，执行 greedy decoding。
- `max_tokens=8`。
- `logprobs=10`，`logprobs_mode="raw_logprobs"`。
- `max_model_len=131072`，容纳 128000 个输入 token 和八个生成 token。

评分器不会再次调用 chat template。生成数据中的完整 `prompt` 和预编码的 `prompt_token_ids` 会直接进入模型。若实际 Tokenizer 添加了额外 special token，或者实际 token ID 与构造记录不一致，程序会在模型推理前终止。

## 结果文件内容

结果文件为：

```text
results/qwen_niah_128k.json
```

结果使用 `schema_version=2`，包含 `run`、`summary` 和 `records`。

每条 `records` 保存：

- `sample_id`、`phase`、`row_index`、`binding_id`、`sentence_pair_id` 和 gold answer。
- 目标 key、目标 value 的文本、token ID 和完整输入 token 位置。
- 实际输入 token 数量、实际输入 token ID hash、构造 prompt token ID hash。
- 生成文本、生成 token ID、每个生成 token 的 logprob 和概率、累计 logprob、结束原因。
- 完整输入最后一个 token 对应的 next-token top 10。每个候选包含 token ID、词表 token、解码文本、rank、raw logprob 和完整 vocabulary 概率。

Qwen 当前没有前导 special token，因此正常记录中：

```text
context_token_position = 127999
predicted_token_position = 128000
```

所有位置从零开始。程序仍然把实际的前导 special token 数量写入记录，并根据实际数量调整目标位置。

`summary` 保存整体准确率、八个 phase 的准确率、正确数量、样本数量、95% Wilson 区间、`r` 与 `r+4` 的准确率绝对差，以及最高 phase 与最低 phase 的差值。

`run` 保存模型目录、model profile、Tokenizer revision 和文件 hash、本地 `config.json` hash、vLLM／PyTorch／Transformers 版本、引擎参数、生成参数、数据配置和 `metadata.json` SHA256。

## 完成检查

评分结束后确认：

1. `results/qwen_niah_128k.json` 存在且可以使用标准 JSON 解析器读取。
2. `schema_version` 为 `2`。
3. `summary.rows` 为 `4096`。
4. `records` 数量为 `4096`，`sample_id` 没有重复。
5. 每条记录的 `completion`、`input.target_key`、`input.target_value` 和 `next_token_distribution.top10` 都存在。
6. 每条 top 10 的 rank 为 1 到 10，概率是有限数值。
7. 八个 phase 的样本数量均为 512。
8. 保存运行日志、完整命令、vLLM 版本、模型路径和结果文件。

完成后把以下文件复制回研究工作区：

```text
results/qwen_niah_128k.json
reports/qwen_niah_validation_cloud.json
```

如果推理过程中断，保留终端输出和 vLLM 日志，记录中断位置和错误信息。不要修改 prompt、Tokenizer、chat template 或配置数量后继续使用同一个结果文件。
