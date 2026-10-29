# NIAH 使用的文本和 Tokenizer

## Paul Graham essays

[niah_128k/paul_graham/filler.txt](niah_128k/paul_graham/filler.txt) 是实际用于补足 prompt 长度的文章文件。它由 49 篇文章按照文件名排序后直接连接，SHA256 为 `b3a70ebc054f2eab5057baf3c4b7e857711472be8086240a516fd29b648ad857`。

`niah_128k/paul_graham/essays/` 保存逐篇文章，`niah_128k/paul_graham/essays.zip` 保存原始下载文件。来源为 `gkamradt/needle-in-a-haystack`，commit 为 `021385d68d3202e37893e9d3cd29011c569abe30`。

论文记录的文章拼接文件 SHA256 为 `51eaf041f492d8fdfdcf87be8756d23888eda0d3cab6816b3c3f11e7682de337`。本次文件与这个 hash 不同，生成数据会保存实际使用的文件 hash。

## 英文句子文件

[niah_128k/sentences/train.txt.gz](niah_128k/sentences/train.txt.gz) 是 `agentlans/high-quality-english-sentences` 的训练文件，revision 为 `5dfcb23b31fa9b85ce90b5713fcbba29179f2401`，SHA256 为 `edbcece9d0236c149f872b1098a1cb952e43a61bfc6cde0c42b44adbeb594406`。

程序从这里挑选 `natural_prefill` 和 `natural_postfill`。原始数据集文件信息保存在 [metadata.json](niah_128k/sentences/metadata.json)。

## Qwen Tokenizer 和官方资料

`qwen3_8_flash_next/` 保存固定 revision `de4b8e4d43b917e7706784d8bb445c9af86a3540` 的 `tokenizer.json`、`tokenizer_config.json`、`chat_template.jinja`、`config.json`、`generation_config.json`，以及官方 model card、technical report 和 Tokenizer 检查记录。

模型权重目录约定为 `/models/Qwen3.8-Flash-Next`。这里保存的是生成和检查 prompt 使用的 Tokenizer 文件及参考资料。

## 文件来源和检查

[manifest.json](manifest.json) 记录全部 63 个资源文件的大小、SHA256 和来源，并列出文章连接顺序。[download_manifest.json](niah_128k/download_manifest.json) 保留原始下载记录，文件检查结果见 [资源检查报告](../reports/resources_validation.json)。

在仓库根目录执行以下命令，可以检查资源文件是否完整，并核对 Tokenizer 的固定版本：

```bash
python -m periodic_weak_spots.niah_128k.check_resources --report reports/resources_validation.json
```

这些资源文件保存在仓库中。完整 prompt 的生成和检查方法见 [Qwen NIAH 使用说明](../docs/qwen_niah_adaptation.zh-CN.md)。
