# Qwen3.8-Flash-Next 的 128K NIAH 使用说明

## 模型目录和 Tokenizer

模型使用 `Qwen/Qwen3.8-Flash-Next`，默认权重目录为 `/models/Qwen3.8-Flash-Next`。在运行推理的机器上，把固定 revision `de4b8e4d43b917e7706784d8bb445c9af86a3540` 的模型下载到这个目录，即可使用注册配置中的默认路径。

[profiles.py](../periodic_weak_spots/niah_128k/profiles.py) 显式注册模型、Tokenizer revision、Tokenizer 文件的 SHA256、默认模型目录以及是否使用官方 chat template。[qwen3_8_flash_next.json](../configs/niah_128k/qwen3_8_flash_next.json) 使用 `model_profile="Qwen/Qwen3.8-Flash-Next"` 和 `dataset_role="evaluation"`。

云服务器执行 prompt 检查、权重下载和评分的步骤见[云端执行说明](qwen_niah_cloud_execution.zh-CN.md)。

生成数据和检查数据只需要 Tokenizer。程序通过 `AutoTokenizer` 读取所需文件；权重在执行 `score` 时由 vLLM 加载。`--tokenizer-path` 可以指定已经下载的 Tokenizer 目录。程序会核查 `tokenizer.json`、`tokenizer_config.json` 和 `chat_template.jinja` 的 SHA256。

仓库保留的 Tokenizer 位于 `resources/qwen3_8_flash_next/`。实际使用的文章、英文句子文件和官方资料见 [资源说明](../resources/README.md)，文件来源和 hash 见 [资源清单](../resources/manifest.json)。

## 模型实际看到的文字

prompt 包含任务说明、填充文章、16,000 条 `key:value。` 记录和最后的提问。Qwen 官方模板还会在外面加入 user、assistant 标记，以及关闭思考的标记。这些标记同样占用 token。

本次配置使用一条 user message，设置 `enable_thinking=False`、`add_generation_prompt=True`。构造器将官方模板生成的完整文字保存到 `prompt`，并把包裹文字保存到 `components.chat_prefix` 和 `components.chat_suffix`。原来的八个组成部分继续单独保存。

完整输入包含这些标记在内，恰好为 128,000 个 token。每条记录有四个 token，目标是从零开始的第 248 条。目标 key 的位置除以 8 的余数决定它属于哪个 group。

在每个 group 中，目标 key 的平均位置为 64,000，位置方差为 28。query 的位置和生成开始位置固定。构造代码位于 [layout.py](../periodic_weak_spots/niah_128k/layout.py)，具体 token 位置保存在每条样本的 `metadata.spans` 中。

本次使用已下载文章生成的数据，各部分长度如下。`L` 按照每个 group 的固定数量表取值，范围为 8–40。

| 部分 | token 数量 |
| --- | --- |
| `chat_prefix` | 3 |
| `intro` | 101 |
| `natural_prefill` | `L` |
| `body_prefix` | 62,880 |
| `pair_body` | 64,000 |
| `natural_postfill` | `48-L` |
| `tail` | 925 |
| `query_separator` | 1 |
| `query` | 33 |
| `chat_suffix` | 9 |

这些数量合计为 128,000。query 从位置 127,958 开始；位置均从零开始编号。

## 补足长度使用的文章

`--filler-file` 指定补足长度用的文章文件，用于 `intro` 的补充文本、`body_prefix` 和 `tail`。代码按照当前 Tokenizer 编码文章，必要时重复文章，再取所需数量的 token。

论文采用标准 NIAH 的 Paul Graham essays，仓库记录的原始文件 SHA256 为：

```text
51eaf041f492d8fdfdcf87be8756d23888eda0d3cab6816b3c3f11e7682de337
```

当前用于检查的文章来自标准 NIAH 公开集合中的 49 篇 Paul Graham essays，按照文件名排序后连接，保存在 `resources/niah_128k/paul_graham/filler.txt`。该文件 SHA256 为：

```text
b3a70ebc054f2eab5057baf3c4b7e857711472be8086240a516fd29b648ad857
```

这份文件与论文记录的文件 SHA256 不同，因此本次构造的 `filler_matches_paper` 为 `false`。逐篇文章、原始下载文件和文章连接顺序均保存在 `resources/`，实际使用的填充文件 hash 还会写入生成数据的 `metadata.json`。

控制位置的 `natural_prefill`、`natural_postfill` 来自另一份英文句子数据集 `agentlans/high-quality-english-sentences`，revision 固定为 `5dfcb23b31fa9b85ce90b5713fcbba29179f2401`。程序使用 Qwen Tokenizer 按长度挑选句子，前后两句合计 48 个 token。两句的顺序会按照固定的数量表交换。

仓库中的 `resources/niah_128k/sentences/train.txt.gz` 是该固定 revision 的原始训练文件。`--source-file` 接受这个文件，读取前会核查 SHA256。省略该参数时，程序从固定版本的数据集加载句子。

## 生成完整数据

在已经安装仓库依赖的环境中执行：

```bash
python -m periodic_weak_spots.niah_128k build \
  --config configs/niah_128k/qwen3_8_flash_next.json \
  --tokenizer-path resources/qwen3_8_flash_next \
  --source-file resources/niah_128k/sentences/train.txt.gz \
  --filler-file resources/niah_128k/paul_graham/filler.txt \
  --output data/niah_qwen_128k
```

以上命令使用仓库中保留的本地文件，并核查文件身份。省略 `--tokenizer-path` 时，程序从固定 revision 加载 Tokenizer；省略 `--source-file` 时，程序从固定版本的数据集加载句子。

每个 group 有 512 条 prompt，共 4,096 条。一个 binding 是包含 16,000 条 key-value 记录的固定有序集合。512 个 binding 在八个 group 中复用，同一个 binding 的 key-value 内容和顺序保持一致。数据使用 4,096 句不同的英文句子组成 2,048 个句子组合，每个组合交换前后角色后使用两次。

输出目录包括 `bindings.jsonl`、`sentence_schedule.jsonl`、`metadata.json`，以及 `rows/key_value/phase_0.jsonl` 到 `phase_7.jsonl`。完整 prompt、各部分文字、答案和 token 位置都可以直接读取。

## 检查生成的 prompt

独立检查入口为 [check_qwen.py](../periodic_weak_spots/niah_128k/check_qwen.py)：

```bash
python -m periodic_weak_spots.niah_128k.check_qwen \
  --dataset data/niah_qwen_128k \
  --tokenizer-path resources/qwen3_8_flash_next \
  --report reports/qwen_niah_validation.json
```

这个命令读取生成数据的配置，核查 Qwen 模型注册信息和 Tokenizer 文件，然后检查全部样本。通过 `--tokenizer-path` 可以使用本地 Tokenizer。

检查内容包括：

- 完整 prompt 恰好为 128,000 个 token，HF 与 GigaToken 对整条文字给出相同的 token ID。
- 官方 non-thinking 模板与保存的完整 prompt 一致。
- 各组成部分分别编码后连接，与完整 prompt 的编码一致。
- 16,000 条记录分别由 key、冒号、value、句号四个 token 组成，全部记录连接后仍然满足要求。
- 每个 binding 的 32,000 个 key/value token ID 互不重复，候选文本能独立编码、解码为相同内容。
- 目标 key、目标 value、答案、query 的位置和记录的 token ID 一致。
- 每个 group 有 512 条样本，每个 binding 在每个 group 中恰好出现一次。
- 句子的文本和 token ID 一致，句子互不重复，前后角色交换关系完整。
- 每条样本的前后两句合计 48 个 token，各组的长度数量符合配置。
- 目标位置的均值为 64,000、方差为 28，固定组件和 query 位置在全部样本中保持一致。
- 完整输入的 token ID hash 与样本记录一致。

[validate.py](../periodic_weak_spots/niah_128k/validate.py) 使用 HF 独立编码检查 Qwen 数据，生成过程使用 GigaToken 编码。发现第一处不符合要求的内容时，程序终止并报告错误。检查成功后，JSON 报告记录总数量、每组位置统计、Tokenizer 文件 hash 和检查环境的版本。

原来的 `validate` 子命令也支持 Qwen 数据，并可以通过 `--report` 保存结果。

## 本次检查结果

完整数据保存在 `.research/niah_qwen_128k_512/`，Tokenizer 保存在 `resources/qwen3_8_flash_next/`。各组统计和检查环境见 [完整检查报告](../reports/qwen_niah_validation.json)。

| 检查内容 | 结果 |
| --- | --- |
| 样本数量 | 4,096，每个 group 512 |
| 完整输入长度 | 每条 128,000 个 token |
| HF 与 GigaToken 的完整编码 | 4,096 条全部一致 |
| 每组目标 key 平均位置 | 64,000 |
| 每组目标 key 位置方差 | 28 |
| 每组目标 value 平均位置 | 64,002 |
| query 起点 | 全部为 127,958 |
| 前导 special token | 默认编码增加的数量为零 |

检查环境为 Python 3.12.3、Transformers 5.20.2、Tokenizers 0.23.3、GigaToken 0.10.0。16 项测试通过，覆盖实际生成样本、答案变化、位置变化、query 变化、记录变化、模板变化、输入 hash 变化、配置检查以及保留资源的完整性。

原始文章、Tokenizer、英文句子文件和官方资料保存在 `resources/`，检查报告保存在 `reports/`。生成数据和运行中间文件保存在 `.research/`，该目录已经加入 `.gitignore`。

## 在推理机器上使用权重

在用于模型推理的机器执行以下下载命令：

```bash
hf download Qwen/Qwen3.8-Flash-Next \
  --revision de4b8e4d43b917e7706784d8bb445c9af86a3540 \
  --local-dir /models/Qwen3.8-Flash-Next
```

推理环境需要使用支持 `qwen4_exp` 和 `raw_logprobs` 的 vLLM 版本。运行评分时通过 `--tensor-parallel-size` 指定实际使用的 GPU 数量，必须通过 `--results` 指定结果 JSON 文件。FP8 权重按照硬件要求增加 `--enable-expert-parallel`，并使用对应的 vLLM recipe 参数。

`score` 从数据配置取得默认模型目录 `/models/Qwen3.8-Flash-Next`，也接受 `--model` 指定另一个本地目录。加载 Qwen 前会检查该目录中的 Tokenizer 文件。

评分器先使用模型目录中的 Tokenizer 对完整 prompt 编码，再通过 vLLM `TokensPrompt` 传入 `prompt_token_ids`，因此模型实际接收的 token 序列在推理开始前已经经过检查。评分使用 `generation_config="vllm"`、`temperature=0`、默认 `max_tokens=8`，按照 `completion.strip().startswith(gold)` 判断答案。八个 group 的准确率和最高、最低准确率之差会保存在结果中。默认 `max_model_len=131072`，可以容纳 128000 个输入 token 和八个生成 token。

## 每条 prompt 的实验记录

结果文件的 `schema_version=2`，包括 `run`、`summary` 和 `records`。`records` 中每条记录保存：

- `sample_id`、`request_id`、`phase`、`row_index`、`binding_id`、`sentence_pair_id`。
- `input.target_key`、`input.target_value`：各自的文本、token ID 和 `token_position`。
- `input.prompt_token_count`、实际输入和构造 prompt 的 token ID hash、`input.query_token_start` 和前导 special token 数量。
- `completion.text`、`completion.token_ids`、每个生成 token 的 logprob 和概率、累计 logprob、`completion.correct`、`completion.finish_reason`、`completion.stop_reason`。
- `next_token_distribution.top10`：每个候选的 `rank`、`token_id`、`vocabulary_token`、`decoded_text`、`logprob` 和 `probability`。

所有 `token_position` 从零开始，采用 vLLM 实际输入的完整 token 序列，包含对话模板和前导 special token。程序根据 vLLM 返回的 `prompt_token_ids` 核查输入长度、token ID hash、目标 key、目标 value 和 query。Qwen 的 token ID 必须与构造记录完全一致。

`next_token_distribution` 记录完整 prompt 最后一个输入位置预测的下一个 token，即第一个生成 token 的分布。`context_token_position` 指定最后一个输入位置，`predicted_token_position` 指定第一个生成位置；本次 Qwen 数据分别为 127,999 和 128,000。

评分器请求 `logprobs=10` 并显式采用 `logprobs_mode="raw_logprobs"`。`probability=exp(logprob)` 使用整个 vocabulary 的 softmax 概率，十个候选之间不再归一化。生成答案继续使用 greedy decoding。缺少完整 top 10、概率无效或实际输入不符合记录时，程序立即报告错误。

`summary` 还保存每组的正确数量、样本数量、95% Wilson 区间，以及 `r` 与 `r+4` 的准确率差。`run` 保存模型目录、model profile、Tokenizer revision 和文件 hash、本地 `config.json` hash、vLLM／PyTorch／Transformers 版本、传入的引擎参数、生成参数，以及完整数据配置和 `metadata.json` 的 SHA256。

输入记录函数已经用全部 4,096 条真实 prompt 完成检查，结果见 [输入记录检查报告](../reports/qwen_prompt_completeness.json)。评分记录相关的 10 项 CPU 测试通过。模型推理记录保存在 `results/` 下的评分结果文件与实时进度文件中。
