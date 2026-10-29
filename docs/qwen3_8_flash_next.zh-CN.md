# Qwen3.8-Flash-Next 架构与官方提示词

## 官方模型与资料版本

本地部署对应的官方模型 ID 为 `Qwen/Qwen3.8-Flash-Next`。官方 model card 将这个 checkpoint 标为经过 post-training 的模型。2026-10-26 核查的 Hugging Face revision 为 `de4b8e4d43b917e7706784d8bb445c9af86a3540`。[官方 model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/README.md)

架构参数来自该 revision 的 `config.json` 和 Qwen 官方 technical report《On the Design of Qwen3.8-Next Architecture: Evaluation, Efficiency, and Training Stability》。该报告发表于 2026-08-26。[官方配置](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json)，[官方 technical report](https://raw.githubusercontent.com/QwenLM/Qwen3.8-Flash-Next/main/tech_report.pdf)

## 语言模型结构

配置中的 `architectures` 为 `Qwen4ExpForConditionalGeneration`，顶层 `model_type` 为 `qwen4_exp`，`text_config.model_type` 为 `qwen4_exp_text`。模型具有 vision encoder；纯文本 NIAH 的输入使用文本 Tokenizer 和语言模型部分。[官方配置](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json)

| 部分 | 官方参数 |
| --- | --- |
| 语言模型参数规模 | 主体约 125B，每个 token 激活约 6B；另有约 51B n-gram embedding 和约 4B MTP |
| 层数与 hidden size | 48 层；`hidden_size=2560` |
| 层排列 | 12 组，每组为三个 Gated DeltaNet layer 和一个 QSA layer；每层还有 MoE |
| MoE | 512 个 routed expert，每个 token 选择 10 个；另有一个 shared expert；intermediate size 为 640 |
| Gated Residual | 四个 residual branch；bottleneck rank 为 320 |
| n-gram embedding | `ngram_size=3`，包含 bigram/trigram lookup，配置的 vocabulary base 为 20,000,000；`ple_layer_ids=[2]` |
| MTP | 一个额外 layer，训练时使用 multi-step prediction |
| 原生 context length | 262,144 个 token |

参数规模采用官方 model card 分别列出的统计。层排列、Gated Residual 与 n-gram embedding 的计算方式见 technical report 第 3–15 页；数值配置见 `text_config`。[官方 model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/README.md)，[官方 technical report](https://raw.githubusercontent.com/QwenLM/Qwen3.8-Flash-Next/main/tech_report.pdf)，[官方配置](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json)

Gated DeltaNet 使用固定大小的 recurrent state 保存历史信息。配置为 16 个 QK head、48 个 value head，每个 head 的维度为 128，前部 causal convolution 的 kernel size 为 4。Gated Residual 为每层提供四个信息通道，并通过输入相关的 gate 控制读取与写入。[官方 technical report，第 3–4、10–14 页](https://raw.githubusercontent.com/QwenLM/Qwen3.8-Flash-Next/main/tech_report.pdf)，[官方配置](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json)

QSA 所在层在 `layer_types` 中标为 `full_attention`。该 checkpoint 在这些层使用 QSA，序列中的层号为 3、7、11，依此类推，直到 47，均从零开始编号。[官方 technical report，第 7 页](https://raw.githubusercontent.com/QwenLM/Qwen3.8-Flash-Next/main/tech_report.pdf)，[官方配置](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json)

## QSA 中的四个 token 分组

QSA 的 lightweight indexer 对连续四个 token 的 key projection 取平均值，得到一个 micro-block 的表示。block 从位置 `4b` 开始，覆盖 `[4b,4b+3]`；完整 block 形成后才能被 query 选择。indexer 在平均之后进行 RMSNorm 和 partial RoPE，RoPE 使用 block 的起始位置。[官方 technical report，第 5–7 页，Equation 12–19](https://raw.githubusercontent.com/QwenLM/Qwen3.8-Flash-Next/main/tech_report.pdf)

| 参数 | 配置值与作用 |
| --- | --- |
| `indexer_compress_ratio` | 4，表示每个 micro-block 的 token 数量 |
| `indexer_budget` | 2,048，表示选择完整 block 时的 token budget |
| block budget | 512 个完整 block，即 `2048/4` |
| `indexer_n_heads` | 四个 query head |
| `indexer_kv_heads` | 一个 shared key head |
| `indexer_head_dim` | 128 |
| core attention | 24 个 query head、两个 KV head，head dimension 为 256 |
| core attention 的 RoPE | 64 个维度，即 `partial_rotary_factor=0.25` |

以上参数来自官方配置；indexer 在其 128 个维度中使用 64 个 RoPE 维度。选择的 block 会展开为原始 token index，core attention 读取这些原始 token 的 K/V；当前未完成 block 中的尾部 token 也会加入计算。[官方配置](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json)，[官方 technical report，第 6–7 页](https://raw.githubusercontent.com/QwenLM/Qwen3.8-Flash-Next/main/tech_report.pdf)

Periodic Weak Spots 中的 DeepSeek CSA 将窗口内容汇聚为 core attention 使用的 compressed K/V entry。Qwen QSA 在 indexer 中压缩 key 表示，选择 block 后使用原始 token 计算 core attention。两种结构都引入固定 token 分组，信息读取过程各有特点。

对于 Qwen NIAH，可以将 `t_K % 4` 作为待检验的位置坐标，保留 `%8` 的八个 group 来观察两个周期。是否出现周期性准确率变化，需要模型推理结果确定。Qwen 同时包含 Gated DeltaNet、QSA 和 n-gram embedding，仅凭一条准确率曲线无法确定具体由哪个部分造成变化；这是结合模型结构得出的实验解释限制。

128K 输入连同短答案处于原生 262,144-token 范围内。固定原生 RoPE 配置即可研究这一长度。官方针对超过原生 context 的任务提供 YaRN 设置，扩展上限可达 1,000,000 个 token。[官方 model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/README.md)

## Tokenizer 文件与编码方式

Tokenizer 使用同一个模型仓库中的 `tokenizer.json`、`tokenizer_config.json` 和 `chat_template.jinja`。`tokenizer.json` 定义 BPE，使用 NFC normalization、正则 pre-tokenization 和 ByteLevel 编码、解码。`add_prefix_space=false`，`add_bos_token=false`，`clean_up_tokenization_spaces=false`。[官方 tokenizer.json](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/de4b8e4d43b917e7706784d8bb445c9af86a3540/tokenizer.json)，[官方 tokenizer_config.json](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/de4b8e4d43b917e7706784d8bb445c9af86a3540/tokenizer_config.json)

`tokenizer_config.json` 声明 `tokenizer_class="Qwen2Tokenizer"`。本次通过 `transformers=5.20.2` 的 `AutoTokenizer.from_pretrained` 加载固定 revision 文件时，得到的 Python class 为 `Qwen3_5Tokenizer`。class 名称受到 Transformers 版本及其加载逻辑影响；复现时需要同时记录文件 revision 和库版本。

三个 vocabulary 数量有各自的用途：

| 数量 | 含义 |
| --- | --- |
| 248,044 | `tokenizer.json` 中基础 BPE vocabulary 的条目数量 |
| 248,077 | 本次加载后的 `len(tokenizer)`，包含 added token |
| 248,320 | `text_config.vocab_size`，模型 embedding 与输出维度使用的 padded vocabulary 大小 |

以上数量由官方文件与实际加载结果核查。遍历候选 token 时可以沿用仓库的 `len(tokenizer)`；模型输出 vocabulary 的维度按照模型配置处理。

与 NIAH 有关的 token ID 如下，均由真实 Tokenizer 编码确认：

| 文本 | token ID |
| --- | --- |
| `:` | 25 |
| `。` | 1,710 |
| `｜` | 158,819 |
| `<|endoftext|>` | 248,044 |
| `<|im_start|>` | 248,045 |
| `<|im_end|>` | 248,046 |
| `<think>` | 248,068 |
| `</think>` | 248,069 |

Tokenizer 的 `eos_token` 为 `<|im_end|>`，`pad_token` 为 `<|endoftext|>`；`generation_config.json` 的结束 token 列表包含 248,046 和 248,044。`<think>` 与 `</think>` 在 added-token 配置中标为 `special=false`。[官方 tokenizer_config.json](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/de4b8e4d43b917e7706784d8bb445c9af86a3540/tokenizer_config.json)，[官方 generation_config.json](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/de4b8e4d43b917e7706784d8bb445c9af86a3540/generation_config.json)

## 官方 chat template 与默认提示内容

官方提示结构由 `chat_template.jinja` 根据 `messages` 和模板参数生成。单轮纯文本调用的主要参数为 `add_generation_prompt`、`enable_thinking`、`reasoning_effort`。模板会对 message content 应用 `trim`，因此 user 文本首尾空白会影响最终渲染结果。[官方 chat_template.jinja](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/de4b8e4d43b917e7706784d8bb445c9af86a3540/chat_template.jinja)

默认启用 thinking，`reasoning_effort` 默认为 `xhigh`。模板会添加以 `Reasoning effort is set to xhigh.` 开头的 system 指令，要求认真思考、验证关键假设并重视答案的正确性。传入用户自己的 system message 时，这段 reasoning 指令与用户提供的内容连接在同一条 system message 中。[官方 chat_template.jinja](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/de4b8e4d43b917e7706784d8bb445c9af86a3540/chat_template.jinja)

`reasoning_effort` 支持 `xhigh`、`medium`、`low`。`medium` 对应空的额外 reasoning 指令；`low` 添加简短思考指令。thinking 开启时，generation prompt 结尾包含 `<think>\n`。单轮 NIAH 的任务文本可以作为一条 user message 传入。

显式设置 `enable_thinking=False` 后，模板在 assistant 开头放入一个已经结束的空 thinking block。以下为 `messages=[{"role":"user","content":"NIAH"}]`、`add_generation_prompt=True` 的实际渲染结果：

```text
<|im_start|>user
NIAH<|im_end|>
<|im_start|>assistant
<think>

</think>

```

最后的 `</think>` 后面有两个换行。这些内容属于模型输入。真实 Tokenizer 对以上完整文本给出 15 个 token，其中 user content `NIAH` 占三个。默认 `xhigh` 配置的同一条消息为 55 个 token；`medium` thinking 配置为 13 个。改变模板参数会改变输入长度和目标的绝对位置。

使用本次核查的 Transformers 版本，可以按以下方式取得完整输入的 token ID：

```python
input_ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": niah_prompt}],
    tokenize=True,
    return_dict=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
```

这里的 `tokenizer` 是 HF Tokenizer；仓库 `FastTokenizer` 对象通过 `.hf` 访问它。将这个输入用于受控实验时，位置统计应当基于 `input_ids` 中的完整序列。

本地兼容 Chat Completions 的服务，官方示例使用以下请求字段控制模板：

```json
{
  "chat_template_kwargs": {
    "enable_thinking": false
  }
}
```

使用 Python client 时，该字段放入 `extra_body`。`preserve_thinking` 控制多轮消息中的历史 thinking 保留，默认启用；当前单轮 NIAH 的输入不包含历史 assistant 消息。[官方 model card 的 API Usage](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/README.md)

## 官方采样建议与 NIAH 评分参数

官方 model card 的通用生成建议如下。[官方 model card 的 Best Practices](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/README.md)

| 模式 | `temperature` | `top_p` | `top_k` | `min_p` | `presence_penalty` | `repetition_penalty` |
| --- | --- | --- | --- | --- | --- | --- |
| Thinking | 1.0 | 0.95 | 20 | 0.0 | 0.0 | 1.0 |
| Non-thinking | 0.7 | 0.80 | 20 | 0.0 | 1.5 | 1.0 |

Periodic Weak Spots 的 128K NIAH 使用 `temperature=0`、`max_tokens=8` 和直接回答。迁移这项实验时，可以使用官方 non-thinking chat template，并在实验中显式采用论文的 greedy decoding。这样，模型输入格式与受控实验的采样设置都有明确记录。

## 本次真实 Tokenizer 核查

核查使用固定 revision 的官方文件，以及 `transformers=5.20.2`、`tokenizers=0.23.3`、`gigatoken=0.10.0`。

- 仓库 `collect_context_stable_token_candidates` 得到 57,078 个候选 token，满足 16,000 组 record 需要的 32,000 个不同 token。
- 对全部候选构造 query 和 `key:value。` 文本，连同官方 `intro`，共核查 114,157 段文本；HF 与 GigaToken 的 token ID 完全一致。
- `｜` 重复 1–64 次均保持逐个 token 编码。
- `animal_name_pool.json` 中 128 个候选词在新 Tokenizer 下全部仍为单个 token；其中与原文件相同的 token ID 数量为零。
- 本次 `leading_special_ids` 返回空集合，即 HF 默认原始文本编码没有添加前导 special token。
- 默认、`medium` 和 non-thinking 三种实际 chat template 渲染结果，均通过 HF 与 GigaToken 编码一致性检查。

完整数据已在 Python 3.12.3 下生成，共 4,096 条 prompt，每个 group 512 条。独立检查确认每条完整输入恰好为 128,000 个 token，HF 与 GigaToken 的完整编码全部一致；八组的目标 key 平均位置均为 64,000、方差均为 28，query 均从位置 127,958 开始。

[完整检查报告](../reports/qwen_niah_validation.json) 保存各组统计和检查环境。生成与检查命令见 [Qwen NIAH 使用说明](qwen_niah_adaptation.zh-CN.md)。模型推理在配置的权重目录对应的推理机器上执行。
