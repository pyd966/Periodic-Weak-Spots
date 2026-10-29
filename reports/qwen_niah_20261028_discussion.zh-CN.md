# Qwen3.8-Flash-Next 上的 Periodic Weak Spots：配置、结果、观察、猜测与下一步

日期：2026-10-28。本文件汇总当前全部讨论，可以独立阅读。数据与分析文件：

| 内容 | 路径 |
| --- | --- |
| Prefix Padding 原始结果（4,096 条） | `results/qwen_niah_128k_fp8_tp4_full.json` |
| In-sequence Padding 原始结果（4,096 条） | `results/qwen_niah_128k_fp8_tp4_in_sequence.json` |
| 运行环境与官方汇总 | `periodic_weak_spots_bundle_metadata/` |
| 相位机制分析 | `reports/qwen_niah_20261028_phase_mechanism.json`，脚本 `.research/analyze_phase_mechanism.py` |
| 错误输出分类 | `reports/qwen_niah_20261028_wrong_answers.json`，脚本 `.research/analyze_wrong_answers.py` |
| key 特征、`｜` 对照、胜出 record 相位、gold 概率 | `reports/qwen_niah_20261028_four_questions.json`、`reports/qwen_niah_20261028_four_questions_followup.json`，脚本 `.research/analyze_four_questions.py`、`.research/analyze_four_questions_followup.py` |
| 第一版报告 | `reports/qwen_niah_20261028_report.zh-CN.md` |

## 1. 实验配置

### 1.1 模型与推理

| 项目 | 设置 |
| --- | --- |
| 模型 | `Qwen/Qwen3.8-Flash-Next`，FP8 checkpoint |
| 推理 | vLLM 0.32.0，4 张 RTX PRO 6000 Blackwell，TP=4，expert parallel |
| 解码 | greedy，`max_tokens=8`，记录第一个生成位置的 top-10 概率 |
| 判定 | `completion.strip().startswith(gold)` |

### 1.2 模型结构中与相位有关的部分

| 项目 | 设置 |
| --- | --- |
| 层数 | 48 层，其中 12 层 QSA（第 3、7、…、47 层），36 层 Gated DeltaNet（causal convolution kernel 4） |
| 其他 | 第 2 层 n-gram embedding（bigram、trigram）；MoE 512 个 expert、top-10 |
| QSA core attention | 读取原始 token 的 K/V，GQA，24 个 query head、2 个 KV head，head dim 256，partial RoPE 64 维，没有 KV 压缩 |
| QSA indexer | 4 个 query head、1 个 key head，dim 128；对每 4 个 token 的 key projection 取算术平均，经 RMSNorm，在 block 起点位置加 partial RoPE；block 分数为 `sum_h ReLU(q_h · k̄_b)`；取 top-512 个 block（2,048 token）展开为原始 token，再加上当前未完成 block 的尾部 token；训练时用 KL 从 max-pooled attention 蒸馏 |
| block 划分（vLLM `qwen4_exp/nvidia/ops/qsa_prepare.py`） | 在 `(position + 1) % 4 == 0` 处结束一个 block，block `b` 覆盖绝对位置 `[4b, 4b+3]`，位置从 chat template 的第一个 token 起计数，与数据中的 `t_K` 坐标一致 |

indexer 的池化是固定平均，没有可学习的 gate；12 个 QSA 层共用同一套 block 划分。

### 1.3 Prompt 布局

每条输入 128,000 token，使用官方 non-thinking chat template。16,000 条 `key:value。` record，每条 4 token，目标为第 249 条（`target_index = 248`），目标 key 的平均位置 64,000、方差 28。组件顺序与长度：chat_prefix 3、intro 101、自然语言前句 `L`、body_prefix 62,880、pair_body 64,000、自然语言后句 `48 − L`、tail 925、分隔符 1、query 33（起点 127,958）、chat_suffix 9。

| protocol | 做法 |
| --- | --- |
| Prefix Padding | 前后两句自然语言合计 48 token，整段 record body 随前句长度 `L` 整体移动，全部 record 与目标同相位 |
| In-sequence Padding | 前后各 8 个 `｜` guard，另有 32 个 `｜` 分散插入 record 之间的间隔，每个间隔最多 4 个；目标 record 之前的 248 个间隔共分到 `L − 8` 个；目标 key 位置与 Prefix Padding 逐条相同，distractor 的相位随累计间隔漂移 |

按目标 key 位置 `t_K mod 8` 分为八个 group，每个 group 512 条，共 4,096 条；八个 group 共用同一批 512 个 binding。两种 protocol 的 4,096 个 `sample_id`、gold 和目标 key 位置逐条一致。

### 1.4 四种相位下目标 record 的 block 组成

record 序列为 `… k_{i-1} : v_{i-1} 。 k_i : v_i 。 k_{i+1} : …`。

| `t_K mod 4` | 包含 `k_i` 的 block | 包含 `v_i` 的 block | key 与 value 是否同一 block |
| --- | --- | --- | --- |
| 0 | `[k_i, :, v_i, 。]` | 同一个 | 是 |
| 1 | `[。, k_i, :, v_i]` | 同一个 | 是 |
| 2 | `[v_{i-1}, 。, k_i, :]` | `[v_i, 。, k_{i+1}, :]` | 否 |
| 3 | `[:, v_{i-1}, 。, k_i]` | `[:, v_i, 。, k_{i+1}]` | 否 |

In-sequence Padding 中若目标 record 之前的间隔含一个 `｜`，相位 3 的 key block 变为 `[v_{i-1}, 。, ｜, k_i]`，含两个时为 `[。, ｜, ｜, k_i]`；value block 不变。全部 4,096 条样本中目标之后的间隔都是 0。

## 2. 实验结果

### 2.1 八个 residue group 的准确率

| `r = t_K mod 8` | `t_K mod 4` | Prefix | In-sequence |
| --- | --- | --- | --- |
| 0 | 0 | 510 / 512（99.61%） | 506 / 512（98.83%） |
| 1 | 1 | 508 / 512（99.22%） | 508 / 512（99.22%） |
| 2 | 2 | 509 / 512（99.41%） | 506 / 512（98.83%） |
| 3 | 3 | 493 / 512（96.29%） | 488 / 512（95.31%） |
| 4 | 0 | 509 / 512（99.41%） | 507 / 512（99.02%） |
| 5 | 1 | 509 / 512（99.41%） | 509 / 512（99.41%） |
| 6 | 2 | 507 / 512（99.02%） | 507 / 512（99.02%） |
| 7 | 3 | 489 / 512（95.51%） | 490 / 512（95.70%） |
| 全部 | | 4034 / 4096（98.49%） | 4021 / 4096（98.17%） |
| 最高减最低 | | 4.10 pp | 4.10 pp |
| `r` 与 `r+4` 的平均绝对差 | | 0.39 pp | 0.24 pp |

两种 protocol 合并、按 `t_K mod 4` 统计（每格 2,048 条）：相位 0、1、2、3 的准确率为 99.22%、99.32%、99.07%、95.70%，相位 3 的 Wilson 95% 区间为 [94.74%, 96.50%]。相位 3 与其余相位的两比例 z 检验：Prefix `z = −7.83`（`p ≈ 5e-15`），In-sequence `z = −7.33`（`p ≈ 2e-13`）。相位 2 与相位 0、1 的差异 `z = −0.63` 和 `z = −0.53`，没有统计意义。

### 2.2 两种 protocol 的配对

两种 protocol 都错 39 条（其中相位 3、7 占 24），只有 Prefix 错 23 条，只有 In-sequence 错 36 条，生成文本逐条相同 4,008 / 4,096。McNemar `χ² = 2.86`，`p ≈ 0.09`。

### 2.3 错误样本的形态

gold token 在 top-10 中的情况（按 `r`）：

| `r` | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Prefix：gold 不在 top-10 | 2 | 2 | 3 | 17 | 1 | 1 | 3 | 19 |
| Prefix：gold 概率 < 0.5 | 3 | 4 | 4 | 26 | 5 | 4 | 5 | 27 |
| In-sequence：gold 不在 top-10 | 2 | 1 | 4 | 16 | 1 | 2 | 3 | 18 |
| In-sequence：gold 概率 < 0.5 | 7 | 5 | 7 | 28 | 6 | 4 | 7 | 27 |

错误样本中 gold 概率的分布（top-10 底部概率约 0.002）：

| | 相位 3 错误 | 其他相位错误 |
| --- | --- | --- |
| Prefix：gold 不在 top-10 / 错误总数 | 36 / 42 | 12 / 20 |
| In-sequence：gold 不在 top-10 / 错误总数 | 34 / 46 | 13 / 29 |

其他相位的错误里约一半是 gold 概率与 top-1 相差不到一倍的险胜型错误，相位 3 的错误绝大多数是 gold 概率低于千分之二。正确样本中也有削弱：gold 概率低于 0.9 的条数，相位 3 为 48（Prefix）和 41（In-sequence），其他相位为 19 到 25。

错误输出的类型（繁简异体视为同一字）：

| 类型 | 含义 | Prefix 相位 3 | Prefix 其他 | In-sequence 相位 3 | In-sequence 其他 |
| --- | --- | --- | --- | --- | --- |
| key_echo | 输出是目标 key 本身或其异体、子串 | 11 | 1 | 5 | 2 |
| similar_key_record | 输出是另一条 record 的 value，该 record 的 key 与目标 key 相近 | 6 | 6 | 15 | 6 |
| gold_variant | 输出是 gold 的异体或子串 | 4 | 4 | 4 | 8 |
| unrelated | 其他 | 21 | 9 | 22 | 13 |

错误输出 top-1 概率的中位数：Prefix 相位 3 为 0.085，In-sequence 相位 3 为 0.177；similar_key_record 类型的 top-1 概率多在 0.5 到 0.97。

### 2.4 同一 binding 在不同相位上的一致性

| 项目 | Prefix | In-sequence |
| --- | --- | --- |
| 出错的 binding 数量 | 31 | 41 |
| 只在相位 3 或 7 出错的 binding | 22 | 28 |
| 在相位 3 和 7 同时出错的 binding | 16 | 11 |
| 八个相位全部出错的 binding | 1（`叁→本站只是为了宣传`，gold 是模型无法复现的 token） | 1（同一个） |

相位 3 内按前句长度 `L` 分层，Prefix 各层准确率为 93.4% 到 100%，没有随 `L` 变化的趋势。

### 2.5 出错 key 的特征

以 512 个 binding 为单位，在相位 3、7 的四个 slot（两种 protocol 各两个）上至少错一次的 binding 有 42 个。各特征区分出错与不出错 binding 的 AUC：key 的 token id 0.64，key 字频均值 0.56，与目标 key 共享汉字的其他 record 数量 0.55，key 字数 0.47。token id 最高的四分之一 binding 弱相位错误率 7.4%，其余三组 2.9% 到 3.9%。按字频分四组，错误率两头高、中间低：6.5%、2.5%、4.3%、3.9%（从最常见字到最少见字）。出错 key 分两类：由高频功能字组成的多字词（`的实践`、`这么多的`、`收到的`、`在思想上`、`为此`、`就地`，各有上千条 record 的 key 与之共享汉字）和罕见 token（`叁`、`効`、`锢`、`長輩`、`ញ`、`ғы`）。非汉字文字的 key（假名 12 个、拉丁 9 个）弱相位错误率约 10%，汉字 key 为 3.9%。

前一条 record 的 value 长度、字频、文字种类、是否与目标 key 共享汉字，在弱相位的出错与正确样本之间没有差别，AUC 在 0.41 到 0.57 之间。

### 2.6 `｜` 对照

相位 3、7 的 1,024 条 In-sequence 样本中，目标 record 紧前面的间隔含 `｜` 的有 69 条（66 条一个、3 条两个；相位 3 占 30 条、相位 7 占 39 条）。

| 目标前的 `｜` | 条数 | 准确率 |
| --- | --- | --- |
| 0 | 955 | 95.2% |
| 1 或 2 | 69 | 100% |

按参考长度分层后，没有 `｜` 的各层准确率在 93.4% 到 96.9% 之间，有 `｜` 的各层全部 100%。与 Prefix 同一 binding、同一相位的配对：

| 相位 3、7 的 In-sequence 样本 | Prefix 错、In-sequence 也错 | Prefix 错、In-sequence 对 | Prefix 对、In-sequence 错 | 两边都对 |
| --- | --- | --- | --- | --- |
| 目标前没有 `｜` | 24 | 13 | 22 | 896 |
| 目标前有 `｜` | 0 | 5 | 0 | 64 |

没有 `｜` 时，Prefix 出错的 binding 在 In-sequence 同一相位上有 65% 继续出错；有 `｜` 的 5 个 Prefix 出错 binding（`转载`、`笑著`、`足り`、`たく`、`务必`）全部变对，按 65% 的重复率计算，这件事的概率约 0.5%。

### 2.7 胜出的其他 record 的相位

In-sequence 中目标处于相位 3 时答出其他 record value 的 38 条样本，胜出 record 的相位分布为 0:13、1:7、2:4、3:14，与均匀分布的卡方值 7.3（自由度 3）。胜出 record 距离目标的 `|offset|` 中位数在 1,900 到 4,500 条 record 之间；紧邻目标的只有 `化纤` 和 `锢` 两个 binding 的固定错误，在各个相位都出现。相位 3 下胜出者中 15/38 与目标 key 共享汉字，其他相位为 6/16。

## 3. 论文的观察与结论

论文为 Zhu et al. 2026（arXiv 2609.36322）。

### 3.1 现象

| 模型 | 现象 |
| --- | --- |
| DeepSeek-V4-Flash-Base | 128K NIAH 最高减最低 40.2 pp，周期 4，弱相位 `r mod 4 ∈ {0, 3}` |
| DeepSeek-V4-Flash（post-trained） | 19.1 pp，周期 4 |
| DeepSeek-V4.1-Flash | 周期 2，与其 stride 一致 |
| DeepSeek-V3.1-Base（无分块压缩） | 无周期 |
| 论文受控模型（Qwen3-0.6B 从头训练，仅加 KV 压缩） | 全部 23 个压缩模型都有周期，Prefix Padding 最高减最低最大 78 pp；3 个 full attention 对照最大 6.1 pp |

受控模型中：周期跟随 stride，与 window 大小无关；去掉 RoPE 仍有周期；把 gate 换成均匀平均后周期仍在，最高减最低从 75.3 pp 降到 19.2 pp。

### 3.2 机制分析

- KV-head knockout（mean replacement）：full attention 模型里每层对所有相位贡献相同；压缩模型里有的 head 只对某一两个相位重要，并每隔一个 stride 重复。
- gate 的静态集中：训练中一部分 head 的 compression gate 变成固定的，key 总取 window 内第 `i` 个位置、value 取第 `i+1` 个，相当于只记一个 bigram；gate 高峰位置与该 head 负责的相位对应；循环平移 gate 参数，准确率曲线随之平移。
- 理论：在简化的 bigram 检索模型里，最优压缩器把 key 的全部权重放在一个位置、value 放在下一个位置（hard adjacent selection），梯度下降从小初始化高概率收敛到这种状态；多个 head 互相错开占据不同相位，但没有机制保证覆盖所有相位。
- 论文把相位效应分为两种不对称：window 内各 offset 之间的差异，由 gate 的静态集中解释；key 与 value 跨 window 边界的差异，平移 gate 参数无法恢复，论文承认没有统一解释。

### 3.3 论文没有覆盖的部分

- 受控模型对全部压缩 entry 做 dense attention，没有 top-k，没有 indexer；全文没有出现 "indexer"。附录 D 的 "DeepSeek-style kernel" 只复现 CSA 的压缩部分（W8/S4、vector gate、K/V 共用、前后两半各一组 payload 和 gate map），lightning indexer 没有被复现。论文证明的是：在所有 entry 都可见的前提下，压缩表示本身足以造成周期。
- 对 DeepSeek-V4-Flash-Base 的干预是替换整层 attention 输出，论文写明它 "do not identify a particular query head, a compressed-only branch, or the compression operation that wrote the relevant information"。论文没有在任何模型上区分"entry 没被选中"和"entry 被选中但信息不足"。
- 没有任何"部分层压缩、部分层读原文"的模型。

### 3.4 三类模型的结构对比

| 项目 | 论文受控模型（W8/S8 参考配置） | DeepSeek-V4-Flash | Qwen3.8-Flash-Next |
| --- | --- | --- | --- |
| core attention 读取的 KV | 压缩 entry：每 8 token 一个，scalar gate + adaptive softmax 池化，另有固定 16 token 的 sliding window | CSA 层：每 4 token 一个 entry，由 8 token 经 softmax gate 加权求和（W8/S4），另有 128 token 的 sliding window；HCA 层每 128 token 一个 entry | 原始 token 的 K/V，没有 KV 压缩 |
| 稀疏选择 / indexer | 无，dense attention | lightning indexer 对压缩 entry 打分，top-512；HCA 层无 indexer | indexer 对 4 token 平均后打分，top-512 个 block，展开为原始 token |
| 固定 token 分组出现的位置 | KV 压缩 | KV 压缩 + indexer（同一分组） | 仅 indexer |
| 使用该机制的层 | 全部层 | 前 2 层 sliding window，其余 CSA 与 HCA 交替，共 43 层 | 48 层中 12 层 QSA，其余 Gated DeltaNet |
| 测得的相位效应 | 最高减最低达 78 pp | Base 40.2 pp，post-trained 19.1 pp，弱相位 `r mod 4 ∈ {0, 3}` | 4.1 pp，弱相位仅 `r mod 4 = 3` |

## 4. 我们的观察

每条都附带证据来源。

1. 周期为 4，弱相位只有 `t_K mod 4 = 3`，两种 protocol 一致（2.1）。4 是 indexer 的 block 大小；Gated DeltaNet、n-gram embedding、MoE 都没有以 4 为模数的位置结构。
2. In-sequence Padding 打散 distractor 相位后，相位 3 的下降幅度不变（2.1、2.2）。下降由目标 record 自身的相位决定。
3. "key 和 value 跨 block"本身不造成下降：相位 2 同样跨 block，准确率与相位 0、1 没有差别（1.4、2.1）。
4. 相位 3 的错误绝大多数是 gold 概率归零，其他相位的错误一半是险胜型；相位 3 的正确样本中低置信度的比例也接近其他相位的两倍（2.3）。
5. 错误由"key 内容 + 相位"决定：同一 binding 在相位 3 和 7 上重复出错，在两种 protocol 之间重复出错，与前句长度无关（2.4）。
6. 出错 key 要么是高频功能字组成的多字词，要么是罕见 token，中间频率的 key 最少出错；单一特征的预测力都弱（2.5）。
7. 目标前面插入 `｜` 后，相位 3 的错误消失，包括 5 个在 Prefix 中出错的 binding；此时 key 仍在 block 末尾、仍占四分之一、value block 完全相同（2.6）。
8. 目标 block 丢失后，模型答出的其他 record 是整个 prompt 范围内 key 最像的一条，这些 record 本身没有相位倾向（2.7）。
9. 前一条 record 的内容特征（长度、字频、文字种类、共享汉字）与相位 3 是否出错无关（2.5）。

## 5. 我们的猜测

### 5.1 三个因素的框架

| 因素 | 含义 | 性质 | 哪些模型有 |
| --- | --- | --- | --- |
| 选择 | top-k 是否把目标 block 选进来 | 二值，余量大（512 取自 32,000），只在最弱的相位上偶尔失手 | Qwen3.8、DeepSeek |
| 表示 | 压缩把若干 token 挤进一个向量，key 或 value 的信息在写入时被挤掉 | 连续，每次检索都付出，gate 的静态集中让各 head 分别放弃不同相位 | 论文受控模型、DeepSeek |
| 弥补通路 | 选中之后能否读到原文 | Qwen3.8 有（core attention 读原始 token），论文模型和 DeepSeek 没有 | |

论文证明"表示"因素单独足够，幅度几十个百分点；我们测到"选择"因素单独存在，幅度约 4 pp；DeepSeek 两者叠加，19 到 40 pp。三个数字来自三个不同的模型，训练数据、规模、后训练都不同，是跨模型比较。"弥补通路"目前没有任何模型能单独检验。

### 5.2 相位 3 的机制假设

状态：与全部观察一致，尚未做因果干预。

- indexer 的 block 分数是 4 个 token 的 indexer key 的平均。key 的身份信息经局部混合向后扩散到紧随其后的 `:`、`v_i`、`。`，不会向前扩散。
- 相位 0 到 2 的 key block 里有两个以上 token 带 `k_i` 的信息，相位 3 只有 `k_i` 自己，另外三个 token 是前一条 record 的 `: v_{i-1} 。`，block 的平均表示被拉向另一条 record。
- top-512 的余量很大，所以"四个 token 带信息"与"两个 token 带信息"的差别不显现，只有最弱的相位 3 碰上身份信号弱的 key 时才跌出前 512。
- `｜` 对照说明起作用的是 key 前面那几个 token 的内容，而非 key 在 block 中的位置：`｜` 不携带内容，只起摊薄作用；前一条 record 的内容会把平均值拉走。
- 失败发生在 key 的 block 这一侧：`｜` 对照中 value block、尾部、query 完全相同，只有 key 的 block 不同，错误就消失。
- 12 个 QSA 层各自独立选择，部分层丢失目标 block 时模型仍能答对但置信度下降，全部层丢失时 gold 概率归零。

保留意见：`｜` 同时改变了 `k_i` 自身的局部上下文（卷积核 4、trigram embedding），现有数据分不开"block 平均被稀释"和"`k_i` 自己的表示变了"。Prefix Padding 中相位 0 到 2 的 `k_i` 前面同样是 `。` 且准确率正常，所以局部上下文本身不致命，它只在 key 处于 block 末尾时才出问题。配对证据建立在 5 个 binding 上。

### 5.3 为什么 Qwen3.8 的周期这样干净

Qwen3.8 的 indexer 没有 gate，block 内没有可以集中的东西，所以没有论文那种 window 内各 offset 的差异；core attention 读原始 token，选中之后信息完整。剩下的只有边界处的选择失败，表现为单一弱相位、小幅度。

### 5.4 对 indexer 设计的启示

- 在 key 之后做平均池化会让 block 末尾的 token 得到最少的上下文支持。可以让 block 表示包含后续少量 token（block-causal 条件改为 `p_b + r + 2 ≤ i` 即可保持因果），或在 token 级打分后按 block 取 max。
- 各 QSA 层使用互相错开的 block 起点，使任意 token 只在四分之一的层中处于 block 末尾。
- 训练时随机平移序列起点，使蒸馏的 teacher 分布覆盖全部相位。
- 若后续实验表明缺的是 key 的 block，上述方案对症；若表明缺的是 value 的 block，对策改为选中 block `b` 时一并选中 `b+1`，或用前移一位的表示打分。

## 6. 下一步能做的实验

### 6.1 只改 prompt，不改模型

1. **前一条 record 的替换实验。** 在 Prefix Padding 的相位 3、7 样本上做三个变体，其余不动：把目标前的 `。` 换成 `｜`；把 `v_{i-1}` 换成 `｜`；把前一条 record 整条换成四个 `｜`。三个变体分别改变 block 里离 key 最近的一个、第二个、全部三个外来 token。第一个变体就消除错误，说明问题集中在紧邻 key 的 token；到第三个变体才消除，说明是整个 block 的平均被前一条 record 占据。规模与现有实验相同，4,096 条。
2. **record 长度为 5 或 6 token。** record 长度等于 block 大小时，相位与 block 组成一一对应；改变 record 长度后对应关系改变，可以检验下降是否始终跟随"key 处于 block 末尾且前面是另一条 record"这一条件。

### 6.2 改 indexer 的选择，不改权重

3. **关闭稀疏选择的对照。** 把 `indexer_budget` 提高到不小于 128,000（全部 block 都被选中），在相同 4,096 条 prompt 上重新评分。相位 3 的下降消失，则下降完全由 block 选择造成。这是整个因果链的第一环。
4. **记录选择结果。** 在每个 QSA 层、最终 query 位置记录 top-512 的 block 集合以及 `k_i` 所在 block 和 `v_i` 所在 block 的分数排名；按相位统计两个 block 各自的选中率，与每条样本的对错交叉；统计全部层都丢失、部分层丢失、没有丢失三种情况下的 gold 概率。仓库里的 oracle indexer 已经能取到 indexer 的 q、k。
5. **强制加入或移除 block。** 在最终 query 位置改写 block 分数，其余不动。四个条件：相位 3 强制加入 key 的 block；相位 3 强制加入 value 的 block；相位 2 强制移除 key 的 block；相位 2 强制移除 value 的 block。哪个条件让相位 3 恢复、哪个让相位 2 下降，就说明 core attention 需要哪个 block。
6. **oracle indexer。** 运行仓库中已实现的 `periodic_weak_spots.niah_128k.oracle_score`：先对每个 token 用 indexer 的 q、k 打分，再按 block 求和。相位 3 恢复，则问题出在"先平均再打分"这一步。
7. **budget 扫描。** budget 从 512 降到 256、128，观察相位 3 的下降是放大还是其他相位也开始下降。这决定 4.1 pp 是否只是余量很大时的尾部效应，以及预算更紧时 indexer 的相位问题会有多严重。

### 6.3 跨模型

8. **DeepSeek-V4-Flash 在相同 prompt 上的结果。** 它同时有压缩和 indexer，弱相位为 `r mod 4 ∈ {0, 3}`。相同 prompt 上对照，可以看相位 3 是否为两个模型共有、相位 0 是否只在有压缩时出现。

### 6.4 优先级

3、4、5 决定结论能否成立；1 是 GPU 空闲前就能准备、改动最小的实验，并直接检验 5.2 的假设；6 和 7 决定结论对 indexer 设计的适用范围；2 和 8 用于确认条件的一般性。
