# Qwen3.8-Flash-Next 128K NIAH 复现报告：Prefix Padding 与 In-sequence Padding

日期：2026-10-28。原始数据：`Periodic-Weak-Spots-experiments-20261028.zip` 中的 `results/qwen_niah_128k_fp8_tp4_full.json`（Prefix Padding）和 `results/qwen_niah_128k_fp8_tp4_in_sequence.json`（In-sequence Padding）。逐条分析结果保存在 `reports/qwen_niah_20261028_phase_mechanism.json` 和 `reports/qwen_niah_20261028_wrong_answers.json`，分析脚本为 `.research/analyze_phase_mechanism.py` 和 `.research/analyze_wrong_answers.py`。

## 1. 实验设置

| 项目 | 设置 |
| --- | --- |
| 模型 | `Qwen/Qwen3.8-Flash-Next`，FP8 checkpoint，vLLM 0.32.0，4 张 RTX PRO 6000 Blackwell，TP=4，expert parallel |
| 输入 | 官方 non-thinking chat template，每条完整输入 128,000 token；16,000 条 `key:value。` record，每条 4 token；目标为第 249 条，目标 key 的平均位置为 64,000，方差为 28 |
| 分组 | 按目标 key 位置 `t_K mod 8` 分为八个 group，每个 group 512 条，共 4,096 条；八个 group 共用同一批 512 个 binding |
| 解码 | greedy，`max_tokens=8`，记录第一个生成位置的 top-10 概率 |
| 判定 | `completion.strip().startswith(gold)` |
| Prefix Padding | 前后两句自然语言合计 48 token，整段 record body 随前句长度 `L` 整体移动，全部 record 与目标同相位 |
| In-sequence Padding | 前后各 8 个 `｜` guard，另有 32 个 `｜` 分散插入 record 之间的间隔（每个间隔最多 4 个）；目标 key 位置与 Prefix Padding 逐条相同，distractor 的相位随累计间隔漂移 |

两组数据的 4,096 个 `sample_id`、gold answer 和目标 key 位置逐条一致，完整 prompt 的 token hash 全部不同。

## 2. 主要结果

### 2.1 八个 residue group 的准确率

| `r = t_K mod 8` | `t_K mod 4` | Prefix 正确数 | Prefix 准确率 | In-sequence 正确数 | In-sequence 准确率 |
| --- | --- | --- | --- | --- | --- |
| 0 | 0 | 510 / 512 | 99.61% | 506 / 512 | 98.83% |
| 1 | 1 | 508 / 512 | 99.22% | 508 / 512 | 99.22% |
| 2 | 2 | 509 / 512 | 99.41% | 506 / 512 | 98.83% |
| 3 | 3 | 493 / 512 | 96.29% | 488 / 512 | 95.31% |
| 4 | 0 | 509 / 512 | 99.41% | 507 / 512 | 99.02% |
| 5 | 1 | 509 / 512 | 99.41% | 509 / 512 | 99.41% |
| 6 | 2 | 507 / 512 | 99.02% | 507 / 512 | 99.02% |
| 7 | 3 | 489 / 512 | 95.51% | 490 / 512 | 95.70% |
| 全部 | | 4034 / 4096 | 98.49% | 4021 / 4096 | 98.17% |
| 最高减最低 | | | 4.10 pp | | 4.10 pp |
| `r` 与 `r+4` 的平均绝对差 | | | 0.39 pp | | 0.24 pp |

两组数据的周期为 4，`r` 与 `r+4` 几乎相等，与 QSA 的 micro-block 大小 4 一致。

### 2.2 按 `t_K mod 4` 合并后的准确率（两种 protocol 合并，每格 2,048 条）

| `t_K mod 4` | 正确数 | 准确率 | 95% Wilson 区间 |
| --- | --- | --- | --- |
| 0 | 2032 / 2048 | 99.22% | [98.73%, 99.52%] |
| 1 | 2034 / 2048 | 99.32% | [98.86%, 99.59%] |
| 2 | 2029 / 2048 | 99.07% | [98.56%, 99.41%] |
| 3 | 1960 / 2048 | 95.70% | [94.74%, 96.50%] |

相位 3 与其余三个相位的两比例 z 检验：Prefix Padding `z = -7.83`（双侧 `p ≈ 5e-15`），In-sequence Padding `z = -7.33`（`p ≈ 2e-13`）。相位 2 与相位 0、1 的差异：`z = -0.63` 和 `z = -0.53`，没有统计意义。

### 2.3 两种 protocol 的配对比较

| 项目 | 数量 |
| --- | --- |
| 两种 protocol 都错 | 39（其中相位 3、7 占 24） |
| 只有 Prefix Padding 错 | 23 |
| 只有 In-sequence Padding 错 | 36 |
| 生成文本逐条相同 | 4008 / 4096 |

McNemar 检验 `χ² = 2.86`，`p ≈ 0.09`。In-sequence Padding 把 distractor 的相位打散后，相位 3、7 的下降幅度不变，说明下降由目标 record 自身的相位决定。

## 3. 相位 3、7 错误样本的特征

### 3.1 gold token 在 top-10 中的分布

| `r` | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Prefix：gold 不在 top-10 | 2 | 2 | 3 | 17 | 1 | 1 | 3 | 19 |
| Prefix：gold 概率 < 0.5 | 3 | 4 | 4 | 26 | 5 | 4 | 5 | 27 |
| In-sequence：gold 不在 top-10 | 2 | 1 | 4 | 16 | 1 | 2 | 3 | 18 |
| In-sequence：gold 概率 < 0.5 | 7 | 5 | 7 | 28 | 6 | 4 | 7 | 27 |

相位 3、7 的多数错误样本中，gold token 完全不在前十名。模型在这些样本上没有读到目标 value。

### 3.2 错误输出的类型

按输出与目标 key、gold、其他 record 的关系分类（繁简异体视为同一字）：

| 类型 | 含义 | Prefix 相位 3 | Prefix 其他相位 | In-sequence 相位 3 | In-sequence 其他相位 |
| --- | --- | --- | --- | --- | --- |
| key_echo | 输出是目标 key 本身或其异体、子串（`長輩→长辈`、`處的→处的`、`务必→務必`、`在思想上→思想上`） | 11 | 1 | 5 | 2 |
| similar_key_record | 输出是另一条 record 的 value，该 record 的 key 与目标 key 相近（`舍弃/抛弃`、`收到的/收到了`、`和信息/和数据`、`年には/日には`） | 6 | 6 | 15 | 6 |
| gold_variant | 输出是 gold 的异体或子串（`預防→予防`、`我不知道→不知道`） | 4 | 4 | 4 | 8 |
| unrelated | 其他 | 21 | 9 | 22 | 13 |
| 合计 | | 42 | 20 | 46 | 29 |

错误输出的 top-1 概率中位数：Prefix 相位 3 为 0.085，In-sequence 相位 3 为 0.177；similar_key_record 类型的 top-1 概率多在 0.5–0.97。相位 3 的错误有两种表现：一种是模型找不到目标 record，以较低置信度复述 key 或给出无关 token；另一种是模型以较高置信度检索到 key 相近的另一条 record。In-sequence Padding 中第二种明显增多（15 对 6），此时 distractor 处于各种相位，相近 key 的 record 若处于相位 0–2 就会在竞争中胜出。

### 3.3 同一 binding 在不同相位的一致性

| 项目 | Prefix | In-sequence |
| --- | --- | --- |
| 出错的 binding 数量 | 31 | 41 |
| 只在相位 3 或 7 出错的 binding | 22 | 28 |
| 在相位 3 和 7 同时出错的 binding | 16 | 11 |
| 八个相位全部出错的 binding | 1（`叁→本站只是为了宣传`，gold 是模型无法复现的长 token） | 1（同一个） |

相位 3 的错误在相位 7 上大量重复，并且在两种 protocol 之间重复（两种 protocol 共同的 39 个错误中 24 个位于相位 3、7）。错误由“key 内容 + 相位”决定，与前句长度 `L` 无关：相位 3 内 `L=11,19,27,35` 四档的准确率为 14/14、195/202、247/258、37/38，没有随 `L` 变化的趋势。Prefix Padding 中相位 3、7 出错 key 的 token id 中位数（134,702）高于全部 512 个目标 key 的中位数（125,224），出错 key 以多字中文词为主（`的实践`、`在思想上`、`这么多的`、`收到的`），其中包含大量高频功能字。

## 4. 对“KV 跨 block”假设的检验

### 4.1 QSA 的 micro-block 划分

vLLM 0.32.0 的 `vllm/models/qwen4_exp/nvidia/ops/qsa_prepare.py` 在 `(position + 1) % 4 == 0` 的位置结束一个 group，`first_position = end_position - 3`；`ops/qsa.py` 对 4 个 raw key 取算术平均（`accumulator / COMPRESS_RATIO`），再经 RMSNorm 和 partial RoPE（位置取 block 起点）；`ops/qsa_indexer.py` 的 block 分数为 `sum_h ReLU(q_h · k̄_b)`，取 top-512 个 block 后展开为原始 token，再加上当前未完成 block 的尾部 token。block `b` 覆盖绝对位置 `[4b, 4b+3]`，位置从 chat template 的第一个 token 起计数，与数据记录的 `t_K` 坐标一致。

### 4.2 四种相位下目标 record 在 block 中的位置

record 序列为 `… k_{i-1} : v_{i-1} 。 k_i : v_i 。 k_{i+1} : …`，目标 key 为 `k_i`，value 为 `v_i = t_K + 2`。

| `t_K mod 4` | 包含 `k_i` 的 block | 包含 `v_i` 的 block | key 与 value 是否同一 block | 准确率（合并） |
| --- | --- | --- | --- | --- |
| 0 | `[k_i, :, v_i, 。]` | 同一个 | 是 | 99.22% |
| 1 | `[。, k_i, :, v_i]` | 同一个 | 是 | 99.32% |
| 2 | `[v_{i-1}, 。, k_i, :]` | `[v_i, 。, k_{i+1}, :]` | 否 | 99.07% |
| 3 | `[:, v_{i-1}, 。, k_i]` | `[:, v_i, 。, k_{i+1}]` | 否 | 95.70% |

相位 2 和相位 3 都把 key 和 value 分到两个 block，但相位 2 的准确率与相位 0、1 没有差别。“key 和 value 跨 block”这个条件本身预测相位 2 也会下降，与数据不符。数据支持的条件更具体：目标 key 是其 micro-block 的最后一个 token。

### 4.3 可以解释相位 3 的机制假设

Qwen3.8 中 QSA 层之前的信息混合以局部为主：Gated DeltaNet 的 causal convolution kernel 为 4，第 2 层的 n-gram embedding 使用 bigram、trigram。key 的身份信息主要保存在 key 自身以及紧随其后的几个 token（`:`、`v_i`、`。`）的 hidden state 中。indexer 对 block 内 4 个 token 的 key projection 取平均：

- 相位 0：block 内 4 个 token 全部位于 `k_i` 之后或等于 `k_i`，平均后的 block 表示包含最多的 `k_i` 信息。
- 相位 3：`k_i` 所在 block 的其余 3 个 token（`:`、`v_{i-1}`、`。`）全部属于前一条 record，平均后 `k_i` 的信息只占四分之一，并被前一条 record 的内容稀释；真正包含 `v_i` 的是下一个 block `[:, v_i, 。, k_{i+1}]`。

这个假设与第 3 节的三个观察一致：gold 不在 top-10（value 所在 block 未被选中，core attention 看不到 `v_i`）、低置信度复述 key（模型只在选中的 token 中看到与 key 相近的内容）、相近 key 的 record 以高置信度胜出（目标 block 的分数被稀释后，其他 block 竞争成功）。这个假设还需要第 6 节的实验确认，当前数据只能说明效应的位置和形态。

## 5. 三类模型的结构对比

| 项目 | 论文受控模型（Qwen3-0.6B-KVC，参考配置） | DeepSeek-V4-Flash | Qwen3.8-Flash-Next |
| --- | --- | --- | --- |
| core attention 读取的 KV | 压缩后的 entry：每 8 个 token 一个 entry（W8/S8），scalar gate + adaptive softmax 池化，另有固定 16 token 的 sliding window | CSA 层：压缩后的 entry，每 4 个 token 一个 entry，每个 entry 由 8 个 token（当前 4 个加前 4 个）经 softmax gate 加权求和（W8/S4），另有 128 token 的 sliding window；HCA 层：每 128 个 token 一个 entry | 原始 token 的 K/V（GQA，24 个 query head、2 个 KV head，head dim 256），没有 KV 压缩 |
| 稀疏选择 / indexer | 无，对全部压缩 entry 做 dense attention | lightning indexer 对压缩后的 entry 打分（indexer key 使用相同的压缩方式），top-512 个 entry；HCA 层无 indexer | 4 个 query head、1 个 key head 的 indexer，对 4 个 token 的 key projection 取平均后打分，top-512 个 block（2,048 token），展开为原始 token |
| 固定 token 分组出现的位置 | KV 压缩（core attention 直接读压缩结果） | KV 压缩 + indexer（二者使用同一分组） | 仅 indexer |
| 使用该机制的层 | 全部层 | 前 2 层为 sliding window attention，其余层 CSA 与 HCA 交替，共 43 层 | 48 层中 12 层为 QSA（第 3、7、…、47 层），其余 36 层为 Gated DeltaNet |
| 位置编码 | RoPE；消融包含 NoPE 和 16 维 RoPE | partial RoPE 64 维；压缩 entry 除可学习位置 bias 外没有额外位置编码 | core attention partial RoPE 64 维；indexer 在平均之后对 block 起点位置加 partial RoPE |
| 论文或本次测得的相位效应 | Prefix Padding 最高减最低达 78 pp，全部压缩模型都有周期性 | Base 模型 40.2 pp，post-trained 模型 19.1 pp，周期 4；弱相位为 `r mod 4 ∈ {0, 3}` | 4.1 pp，周期 4，弱相位仅 `r mod 4 = 3` |

DeepSeek-V4 的数值来自 DeepSeek-V4 技术报告（arXiv 2606.19348）和论文 Table 7；Qwen3.8 的数值来自 `config.json` 和 Qwen3.8-Next 技术报告第 5–7 页。

## 6. 周期现象来源的初步判断

1. Qwen3.8-Flash-Next 的 core attention 没有 KV 压缩，模型中唯一以 4 为周期的结构是 QSA indexer 的 4-token 平均池化（Gated DeltaNet、n-gram embedding、MoE 都没有以 4 为模数的位置结构）。本次观察到的周期 4 效应只能来自 indexer 的 block 选择。
2. 论文的受控模型只有 KV 压缩、没有 indexer，却表现出几十个百分点的周期效应。KV 压缩本身足以产生强周期效应。
3. DeepSeek-V4 的 CSA 同时包含 KV 压缩和在压缩 entry 上工作的 indexer，两者使用同一分组。Qwen3.8 只保留 indexer 这一部分，效应幅度为 4.1 pp，约为 DeepSeek post-trained 模型的四分之一、Base 模型的十分之一。
4. 三个模型的训练数据、规模和后训练都不同，以上是跨模型比较。在同一模型内区分“indexer 选择”与“压缩表示”两个来源，需要第 7 节的实验。

对 indexer 设计的初步启示：

- 在 key 之后做平均池化会让 block 末尾的 token 得到最少的上下文支持。可考虑让 indexer 的 block 表示包含后续少量 token（block-causal 条件改为 `p_b + r + 2 ≤ i` 即可保持因果），或在 token 级打分后按 block 取 max。
- 各 QSA 层使用互相错开的 block 起点（例如第 3 层从位置 0 起，第 7 层从位置 1 起），使任意 token 只在四分之一的层中处于 block 末尾。
- 训练时随机平移序列起点，使 distillation 的 teacher 分布覆盖全部相位。

## 7. 下一步实验（按决定性排序）

1. 关闭稀疏选择的对照：将 `indexer_budget` 提高到不小于 128,000（block budget 不小于 32,000，即全部 block 都被选中），在相同 4,096 条 prompt 上重新评分。若相位 3、7 的下降消失，则下降完全由 block 选择造成。
2. 记录选择结果：在每个 QSA 层记录最终 query token 选中的 512 个 block，统计 `k_i` 所在 block 和 `v_i` 所在 block 在各相位下的选中率，以及目标 block 分数在全部 block 中的排名。这能直接检验第 4.3 节的稀释假设。
3. 运行仓库中已经实现的 oracle indexer（`periodic_weak_spots.niah_128k.oracle_score`，先对每个 token 用 indexer 的 q、k 打分，再按 block 求和）。若相位 3 的下降消失，则下降来自“先平均再打分”这一步。
4. 在相同 prompt 上评测一个 core attention 使用压缩 KV 的模型（DeepSeek-V4-Flash），与 Qwen3.8 的 4.1 pp 对照，量化压缩与选择两种来源的相对大小。
