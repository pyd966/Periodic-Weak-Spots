# NIAH prompt 与官方实验配置

## 128K prompt 的八个组成部分

论文 Appendix C.1（[论文 PDF](<../paper/Zhu 等 - 2026 - Periodic Weak Spots Phase Sensitivity from Chunked KV-Cache Compression.pdf#page=38>)，第 38–40 页）对应 `periodic_weak_spots/niah_128k/`。每条 prompt 按照以下顺序连接八个部分。

| 顺序 | `components` 字段 | 内容与用途 | 生成位置 |
| --- | --- | --- | --- |
| 1 | `intro` | 英文任务说明、三个格式示例，以及使长度满足 `alignment=8` 的补充文本 | [layout.py](../periodic_weak_spots/niah_128k/layout.py) 的 `render_intro`、`_aligned_intro` |
| 2 | `natural_prefill` | 一句自然语言，长度 `L` 为 8–40 个 token，控制目标 key 的位置余数 | [source.py](../periodic_weak_spots/niah_128k/source.py) 的 `curate_sentence_bins`；[schedule.py](../periodic_weak_spots/niah_128k/schedule.py) 的 `build_prefill_schedule` |
| 3 | `body_prefix` | 约 63,000 个 token 的固定填充文本，使目标接近第 64,000 个 token；两端使用 `｜` | `layout.py` 的 `prepare_scaffold`、`_guarded_padding` |
| 4 | `pair_body` | 连续的 16,000 条 `key:value。` record，总计 64,000 个 token | [bindings.py](../periodic_weak_spots/niah_128k/bindings.py) 的 `generate_bindings`；`layout.py` 的 `render_pair`、`validate_pair_body` |
| 5 | `natural_postfill` | 另一句自然语言，长度为 `48-L`；与 `natural_prefill` 合计 48 个 token | `build_prefill_schedule` |
| 6 | `tail` | 固定长度的填充文本，使整条 prompt 达到 128,000 个 token；两端使用 `｜` | `prepare_scaffold`、`_guarded_padding` |
| 7 | `query_separator` | `"\n\n"` | [evaluation.json](../configs/niah_128k/evaluation.json) 的 `layout.query_separator` |
| 8 | `query` | 包含目标 key 两次的英文提问，结尾等待生成 value | `layout.py` 的 `render_query` |

最终连接由 `build_prompt_row` 完成。[build.py](../periodic_weak_spots/niah_128k/build.py) 的 `publish_dataset` 为全部样本复用同一个 `Scaffold`，其中包含固定的 `intro`、`body_prefix` 和 `tail`。

Qwen 专用配置在这八个部分之外加入 `chat_prefix`、`chat_suffix`，将官方 user、assistant 标记计入完整输入长度。构造和检查方法见 [Qwen 128K NIAH 使用说明](qwen_niah_adaptation.zh-CN.md)。

`render_intro` 中三个格式示例为 `猫:树。`、`狗:叶。`、`鸟:根。`。它们说明 record 格式。实际检索的 16,000 条 record 来自 `Binding.pairs`。

`render_query` 的完整格式字符串如下，其中 `{key}` 在两个位置替换为同一个目标 key：

```python
f"Now answer the question using the key-value pairs above. What is the value for key {key}? Answer with only the value. The value for key {key} is "
```

代码中的 `QUERY_SUFFIX = " is "` 包含结尾空格。prompt 在这里结束，gold value 单独保存在 `answer` 中。

自然语言句子来自 `agentlans/high-quality-english-sentences`，固定 revision 为 `5dfcb23b31fa9b85ce90b5713fcbba29179f2401`。`body_prefix`、`tail` 和 `intro` 的补充文本来自调用者提供的 `--filler-file`。论文采用 Paul Graham essay haystack；仓库 README 提供其 SHA256，原始文件需要单独取得。

## key 与 value 的 token 结构

128K 构造器使用 `deepseek-ai/DeepSeek-V4-Flash`，revision 为 `60d8d70770c6776ff598c94bb586a859a38244f1`。[tokenization.py](../periodic_weak_spots/tokenization.py) 使用 HF 加载 Tokenizer，通过 GigaToken 编码，通过 HF 解码。构造阶段设置 `add_special_tokens=False`。

`collect_context_stable_token_candidates` 遍历 Tokenizer vocabulary，要求候选项同时满足：

- token ID 至少为 10,000。
- 解码文本通过 `str.isalpha()`，长度最多为 16 个 Unicode 字符。
- 单独编码、解码再编码都保持同一个 token ID。
- 放入 `。{key}:`、`｜{key}:`、`:{value}。` 后，字段边界保持可分别编码。
- 放入完整 query 的两个 key 位置后，仍然分别占用一个 token。

`str.isalpha()` 接受多种文字系统中的字母和汉字。这个筛选条件允许的文本范围包含英文以外的文字。

`generate_bindings` 为每个 binding 从候选集合中无放回抽取 32,000 个 token：前 16,000 个作为 key，后 16,000 个作为 value。因此，一条 prompt 的 record body 中，全部 key 和 value 互不重复。

每条 record 的编码严格为四个 token：`key`、`:`、`value`、`。`。目标 record 使用从零开始的 `target_index=248`，即第 249 条；目标 value 位于目标 key 后两个 token。`validate_pair_body` 同时检查单条 record 和全部 record 连接后的编码。

## 位置余数与 compression phase

`phase` 字段在 128K 数据中保存 `r = t_K % 8`，其中 `t_K` 为目标 key 从零开始的 token 位置。模型 compression stride 为 `S` 时，模型对应的 phase 为 `t_K % S`。

DeepSeek-V4 的 CSA 使用 `W=8, S=4`，八个 residue group 覆盖两个 stride 周期。DeepSeek-V4.1-Flash 的 stride 为 2，同样的八个 group 覆盖四个周期。

设 `intro` 长度为 `I`，`body_prefix` 长度为 `B`，自然语言前缀长度为 `L`。代码采用：

```text
B = 64000 - I - 248 × 4 - 24
t_K = I + L + B + 248 × 4 = 63976 + L
t_V = t_K + 2
```

因此 `L % 8 = r` 就能指定目标 key 的余数。每个 group 的目标 key 平均位置为 64,000，位置方差为 28。目标 value 平均位置为 64,002。改变前后两句自然语言的长度分配，record body 会整体移动；query 的绝对位置与 prompt 总长度保持固定。

`target_to_query_distance` 使用 `query_token_start - target_value_token_start` 计算，距离的起点是目标 value。

评分器 [score_vllm.py](../periodic_weak_spots/niah_128k/score_vllm.py) 将原始文本交给 vLLM，由其应用 Tokenizer 默认的 special-token 行为。数据构造器记录的位置采用未添加 special token 的文本坐标；研究模型实际输入位置时，还需要计入推理端添加的前导 token。

## 128K 实验包含多少条 prompt

默认 [evaluation.json](../configs/niah_128k/evaluation.json) 的组合为：

```text
1 种 pair_form × 8 个 residue group × 每个 group 256 个 binding = 2048 条 prompt
```

`pair_form` 为 `key_value`。一个 binding 是包含 16,000 条 record 的固定有序集合。同一个 binding 在每个 residue group 中使用一次，因此每个 binding 对应八条 prompt。不同 group 中的 `row_index` 与 `binding_id` 存在由 schedule 指定的映射，配对分析应当使用 `binding_id`。

句子配置与 binding 配置通过同一份 schedule 共同指定：

- 2,048 句不同的自然语言组成 1,024 个句子组合。
- 每个组合使用两次，前后句子的角色交换；每个句子总共使用两次。
- 交换后的余数为 `(-r) % 8`，对应关系记录在 `role_reversal_phase` 和 `role_reversal_row_index`。
- 每个 group 包含全部 256 个 binding；句子组合与 binding 之间的排列由 `schedule_seed` 决定。

这些维度通过 schedule 连接。每个 binding 在每个 group 中只分配一组前后句子。

前缀长度也有固定的有限样本数量表。下表单元格的格式为“长度：样本数量”，来自 `FINITE_HISTOGRAM`。

| `r` | `L` 的样本数量 |
| --- | --- |
| 0 | 8：1；16：52；24：150；32：52；40：1 |
| 1 | 9：2；17：68；25：146；33：40 |
| 2 | 10：4；18：84；26：140；34：28 |
| 3 | 11：7；19：101；27：129；35：19 |
| 4 | 12：12；20：116；28：116；36：12 |
| 5 | 13：19；21：129；29：101；37：7 |
| 6 | 14：28；22：140；30：84；38：4 |
| 7 | 15：40；23：146；31：68；39：2 |

每个 group 都有恰好 256 条，`mean(L)=24`、`mean(L²)=604`、`Var(L)=28`。这些数值在生成后的有限数据集中精确成立。

| Table 7 中的模型 | 每个 residue group | 每个模型的 prompt 数量 |
| --- | --- | --- |
| `DeepSeek-V4-Flash-Base` | 256 | 2,048 |
| `DeepSeek-V4-Flash-0731` | 256 | 2,048 |
| `DeepSeek-V4-Pro-Base` | 256 | 2,048 |
| `DeepSeek-V4-Pro-0813` | 256 | 2,048 |
| `DeepSeek-V4.1-Flash` | 2,560 | 20,480 |

前四个模型共享同一套 2,048 条 prompt 和 sample ID。V4.1 使用相同分布，binding 和每个 group 的样本数量扩大十倍。公开构造器支持通过 256 的整数倍扩大 `rows_per_phase`，同时要求 `sentence_pair_count=4×rows_per_phase`、`distinct_sentence_count=8×rows_per_phase`。

公开行为评分命令对每条 prompt 生成一次 completion，参数为 `temperature=0`、`max_tokens=8`。配置中没有“对同一条 prompt 重复生成多次”的维度。

## evaluation 与 calibration 的配置

128K 的 [calibration.json](../configs/niah_128k/calibration.json) 具有相同的长度、record 数量和 residue 设计，使用以下独立 seed 配置：

| 参数 | evaluation | calibration |
| --- | --- | --- |
| `source.shuffle_seed` | 20260720 | 20260902 |
| `phase_design.schedule_seed` | 20260721 | 20260903 |
| `binding_seed` | 20260625 | 20260904 |

`calibration` 用于估计因果干预需要的平均 activation。普通行为评分读取 `evaluation`。两个配置共用句子来源与 token 候选集合；seed 区分了生成过程。需要确保 calibration 与 evaluation 文本互不重复的干预过程，还会检查 prompt digest。

## 128K 每条样本保存的信息

`publish_dataset` 生成以下文件：

| 文件 | 保存内容 |
| --- | --- |
| `metadata.json` | 完整配置、Tokenizer revision、句子筛选报告、长度分布与位置统计、文件路径及 matrix 大小 |
| `bindings.jsonl` | 每个 binding 的全部 key、value、token ID、目标 index、目标 key 与目标 value |
| `sentence_schedule.jsonl` | 每条样本使用的 binding、前后句子文本及 token ID、来源 index、句子角色交换关系 |
| `rows/key_value/phase_0.jsonl` 至 `phase_7.jsonl` | 每个 residue group 的完整 prompt、答案和位置记录 |

每条 prompt row 包含：

- 标识与分组：`sample_id`、`pair_form`、`phase`、`row_index`、`binding_id`。
- 输入与答案：`prompt`、`answer`、八个部分的 `components` 文本。
- 句子关系：`sentence_pair_id`、`prefill_sentence_id`、`postfill_sentence_id`、`role_reversal_phase`、`role_reversal_row_index`。
- 长度控制：`prompt_token_count`、`model_prefill_token_count`、`phase_control_prefix_count`、`fixed_natural_filler_count`、`pair_token_count`。
- 目标与 query：`target_index`、`target_pair_token_start`、`target_value_token_start`、`target_phase`、`query_token_start`、`query_token_count`、`target_to_query_distance`。
- 各组成部分的位置：`spans` 中的 `start` 和 `count`。

长度控制、目标与 query、各组成部分的位置这三组字段位于 row 的 `metadata` 中。`model_prefill_token_count` 当前与文本构造长度相同，值为 128,000。

`build` 命令在标准输出打印 `filler_sha256`、`filler_matches_paper`，并将实际使用的 `filler_sha256` 写入 `metadata.json`。

独立 [validate.py](../periodic_weak_spots/niah_128k/validate.py) 会重新编码完整文本，并检查 record、组件边界、位置、答案、长度直方图和 schedule 关系。`build` 内部包含 token 检查；独立验证需要运行 `validate` 子命令。

## 128K 评分保存的信息

正确性由 `is_correct` 定义：

```python
completion.strip().startswith(gold)
```

因此，gold 后面继续出现标点或其他文本，也可以计为正确。评分结果字段为：

- `rows`：参与统计的 prompt 数量。
- `answer_prefix_accuracy`：整体准确率。
- `answer_prefix_accuracy_by_residue`：八个 residue group 的准确率。
- `gap`：最高 group 准确率减去最低 group 准确率。

准确率与 `gap` 都使用 0–1 的数值，转换成百分比或百分点时乘以 100。`score` 必须通过 `--results` 指定结果文件；整体统计保存在 `summary`。

结果文件使用 `schema_version=2`，包含 `run`、`summary` 和逐条 `records`。每条记录包含目标 key、value 的文本、token ID 和实际输入 token 位置，完整输入的长度和 token ID hash，以及生成文本、生成 token ID、逐 token logprob、累计 logprob、正确性和结束原因。完整输入最后一个位置的 next-token top 10 同时保存 token、rank、raw logprob 和整个 vocabulary 下的概率。具体字段见 [每条 prompt 的实验记录](qwen_niah_adaptation.zh-CN.md#每条-prompt-的实验记录)。

`summary` 保存每个 residue 的正确数量、样本数量、95% Wilson 区间、`r` 与 `r+4` 的准确率绝对差，以及最高组和最低组的准确率差。

`run` 保存模型目录、Tokenizer revision、Tokenizer 文件 hash、模型 `config.json` hash、推理库版本、传入的引擎参数、生成参数和数据配置。

## Qwen3 受控模型的 prompt 组成

论文 Appendix C.2–C.3（第 40–43 页）对应 `periodic_weak_spots/niah_0p6b/`。这里测试作者基于 Qwen3-0.6B 从头训练的模型，Tokenizer 固定为 `Qwen/Qwen3-0.6B`，revision 为 `9d4bfd9a94aa5f2ab18d77fa457c306da0b8e439`。

| 顺序 | 内容 | 配置与生成位置 |
| --- | --- | --- |
| 1 | `Remember each animal's assigned human name.\n` | 配置中的 `template.task_descriptor` |
| 2 | Prefix Padding 的自然语言前缀，或者 In-sequence Padding 的固定前部 guard | [build.py](../periodic_weak_spots/niah_0p6b/build.py) 的 `assemble_episode` |
| 3 | ` Animal-name pairs:\n` | `template.mapping_start` |
| 4 | 64 条 animal-name record；In-sequence Padding 在 record 之间插入 spacer | `template.pair_template`、`render_template`、`assemble_episode` |
| 5 | 自然语言后缀，或者固定后部 guard | `assemble_episode` |
| 6 | ` Recall the requested animal names:\n` | `template.recall_query_prefix` |
| 7 | 八个已有 animal-name mapping 的示例 | `_demonstrations_for_sample`、`_render_query` |
| 8 | 目标 animal key；在这里结束输入 | `template.target_template`、`_render_query` |

record 与示例均使用 `"{key}{value}\n"`。候选词的文本自带前导空格，例如 `" bear"` 和 `" Susan"`，连接后为 `" bear Susan\n"`。一条 record 包含 animal、name、换行三个 token，目标 value 紧跟目标 key。

目标 record 的 `target_index=24`，即第 25 条。八个示例从其余 63 条 record 中选择，排除目标 record。输入结尾只包含目标 animal；gold name 保留在评分字段中。

候选词和原始 token ID 保存在 [animal_name_pool.json](../configs/niah_0p6b/animal_name_pool.json)。[template.py](../periodic_weak_spots/niah_0p6b/template.py) 的 `derive_template_geometry` 计算各部分 token 数量，`render_template` 检查文本连接后的编码。

## 两种 padding protocol 的配置

每个 protocol 的 evaluation 都包含：

```text
1000 个 logical example × 24 个 source-key residue = 24000 条 prompt
```

一个 logical example 固定 animal-name mapping、record 顺序和八个示例的 index。它在全部 24 个 residue 下复用这些内容。`strides=[3,4,6,8,12]` 的最小公倍数为 24；实际模型 stride 为 `S` 时，按照 `r % S` 合并 residue group。

两个 protocol 共用 seed `20260722`，并通过相同的 sample index 重建相同的 mapping 与示例。它们共计形成 48,000 条 evaluation prompt。对应 calibration 使用 logical example index `[1000,2000)`，每个 protocol 另外包含 24,000 条；evaluation 使用 `[0,1000)`。

| protocol | 位置控制方式 | 固定的插入 token 数量 | 配置 |
| --- | --- | --- | --- |
| Prefix Padding | 前后各插入一句完整自然语言；`Lpre+Lpost=144` | 144 | [prefix_padding.json](../configs/niah_0p6b/prefix_padding.json) |
| In-sequence Padding | 前后各有 24 个 `｜`，另外 126 个 `｜` 分配到 63 个 record 间隔，每个间隔最多四个 | 174 | [in_sequence_padding.json](../configs/niah_0p6b/in_sequence_padding.json) |

Prefix Padding 中，`Lpre` 与 `Lpost` 均在 24–120 之间。每个 residue 下长度分布的期望为 72、方差为 209。有限样本由随机抽样产生，各 group 的样本均值允许不同。128K 的长度表则在有限数据中精确匹配前两阶统计量。

In-sequence Padding 采用相同的参考长度分布，目标前面 24 个间隔的 spacer 总数为 `Lpre-24`，其余 39 个间隔补足 126。因此，两种 protocol 中目标 key 的条件位置分布相同；In-sequence Padding 的 query 比 Prefix Padding 晚 30 个 token。

句子筛选由 `curate_capped_sentence_bins` 完成。Prefix Padding 的每句文本包含结尾 `"\n\n"`，每个长度最多保存 64 句，并排除包含 animal-name pool token 的句子。In-sequence Padding 使用固定 guard 与 spacer。

论文 Table 8 报告 26 个训练 run，包括 23 个 compressed run 和三个 full-attention baseline。每个模型在两个 protocol 上都有 24,000 条 clean evaluation prompt；训练配置、模型 seed 和干预条件会增加模型计算次数。重复使用同一套 prompt，可以保持模型间的输入条件一致。

## Qwen3 受控实验保存的信息

`build_suite` 把 [EpisodeRow](../periodic_weak_spots/niah_0p6b/config.py) 保存为 HF Dataset，并写入 `manifest.json`。row 中保存：

- `sample_id`、`logical_sample_id`、`phase`。
- 完整 `token_ids`、64 个 `candidate_value_token_ids`、`gold_candidate_index`、`gold_key_token_id`、`gold_value_token_id`。
- `demonstration_pair_indices`。
- `pair_region_start`、`target_pair_start`、`target_value_position`、`postfill_start`、`query_start`、`scoring_position`。
- `prefill_length`、`postfill_length`、两句 filler 的 ID，以及 `gap_extra_lengths`。

`manifest.json` 保存 Tokenizer revision、模板、数量、sample index 范围、stride、query/scoring residue、record 长度以及序列长度范围。输入文本可以通过 token ID 解码获得。

[score_hf.py](../periodic_weak_spots/niah_0p6b/score_hf.py) 调用模型的 `last_token_logits`。`full_vocab_accuracy` 在整个 vocabulary 中选择最高 logit；`candidate_accuracy` 在当前 prompt 的 64 个 name 中选择最高 logit。CLI 的 `--output` 保存整体统计、24 个 residue 的两类准确率，以及 `full_vocab_gap`。逐条两类正确性在 `score_rows` 内部计算，CLI 当前保存汇总结果。

论文 Appendix C.2 提到记录 gold-token probability 和 NLL，Appendix E.3 还提到 gold logit、candidate-set probability。公开基础 scorer 的返回值是两类正确性；研究这些概率指标需要扩展 scorer 的记录内容。

因果干预使用额外输出：`head-sweep` 保存 layer/KV-head 对应的准确率变化；DeepSeek 的 `suffix-sweep` 保存 clean、相同 batch 条件的 clean 和各层各 mask 的 `top1`、`gold_logit`、`gold_nll`；`gate-cycling` 保存不同 `delta` 的 phase 准确率和 `translation_r2`。相关入口位于 [interventions/__main__.py](../periodic_weak_spots/interventions/__main__.py)。

官方发布代码的核查 commit 为 `09c109910ba29b834f54441d96b442ab9e1634ae`。Qwen3.8 的模型资料见 [Qwen3.8 架构与官方提示词](qwen3_8_flash_next.zh-CN.md)，生成和检查命令见 [Qwen NIAH 使用说明](qwen_niah_adaptation.zh-CN.md)。
