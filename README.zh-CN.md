# CaTMeR

[English](README.md)

本仓库提供《CaTMeR：基于候选条件时序记忆的情感与意图感知表情包检索》的训练代码。CaTMeR 基于已被 CEI 2026 接收的会议论文《MGTM: Intent- and Emotion-Guided Temporal Memory for Cross-Session Sticker Retrieval》的配套实现开发。[CEI 2026 会议版代码](https://github.com/ccjy16/MGTM)

CaTMeR 是候选条件时序记忆检索模型（Candidate-Conditioned Temporal Memory for Retrieval）。模型从对话历史中筛选文本并编码记录的相对顺序，再让每个候选表情包分别读取所选记录，以修正各自的检索分数。文本由 Granite Embedding 311M Multilingual R2 编码，表情包的全局和局部视觉特征由 PVT v2 B2 提取。MultiChat 使用意图标签监督训练，DSTC10-MOD 的中英文版本均使用情感标签监督训练。

## 预训练权重

从指定版本下载 Granite 的五个文件，并下载官方在 ImageNet 上预训练的 PVT v2 B2 权重。下文的训练命令使用以下目录结构。 [Granite Embedding 311M Multilingual R2](https://huggingface.co/ibm-granite/granite-embedding-311m-multilingual-r2/tree/44399559930365213510b1ee2eb15ded83374f0e), [PVT v2 分类权重](https://github.com/whai362/PVT/tree/v2/classification)

```text
pretrained_weights/
|-- granite-embedding-311m-multilingual-r2/
|   |-- config.json
|   |-- tokenizer.json
|   |-- tokenizer_config.json
|   |-- special_tokens_map.json
|   `-- model.safetensors
`-- PVT_v2_b2.pth
```

## 数据目录

请遵循各数据集的使用条款获取数据。下面的命令使用仓库根目录下的 `data/`。如果数据存放在其他位置，请在命令中指定实际路径。

### MultiChat

使用由 MultiChat 五个 mode 合并而成的预处理配置 `all`。每个 `pairs/` 和 `*_imgs/` 目录都是一个 LMDB 数据库。每条配对记录都保留了所属的 mode。

```text
data/MultiChat/
|-- all/intent.txt
|-- boba_sample_intent_style_attribute.json
|-- kuaile_sample_intent_style_attribute.json
|-- quanguo_sample_intent_style_attribute.json
|-- siban_sample_intent_style_attribute.json
|-- yongyuan_sample_intent_style_attribute.json
`-- lmdb_all_intent_style_attribute/
    |-- train/pairs/
    |-- val/pairs/
    |-- test/pairs/
    |-- train_imgs/
    |-- val_imgs/
    `-- test_imgs/
```

### DSTC10-MOD

分别准备中文和英文数据。两个语言目录的结构相同。`sticker_emotion_top2_train.json` 为每个表情包 ID 记录最多两个情感标签，标签来自对应语言的训练集。

```text
data/DSTC10-MOD/
|-- zh/
|   |-- dialogues.jsonl
|   |-- splits/
|   |   |-- train.jsonl
|   |   |-- val.jsonl
|   |   `-- test.jsonl
|   |-- images/
|   |   `-- <sticker_id>.png
|   `-- labels/
|       `-- sticker_emotion_top2_train.json
`-- en/
    `-- 目录结构与 zh/ 相同
```

## 训练

在仓库根目录运行以下命令。训练默认使用第一张可见的 GPU。可以通过 `CUDA_VISIBLE_DEVICES` 或 `--gpu-id` 选择设备，`--gpu-id` 按当前可见设备列表编号。

### MultiChat

```bash
python train.py multichat \
  --data-root data/MultiChat/lmdb_all_intent_style_attribute \
  --multichat-annotations-root data/MultiChat \
  --granite-text-model-path pretrained_weights/granite-embedding-311m-multilingual-r2 \
  --pvt-checkpoint pretrained_weights/PVT_v2_b2.pth
```

### DSTC10-MOD-zh

```bash
python train.py dstc-zh \
  --data-root data/DSTC10-MOD/zh \
  --granite-text-model-path pretrained_weights/granite-embedding-311m-multilingual-r2 \
  --pvt-checkpoint pretrained_weights/PVT_v2_b2.pth
```

### DSTC10-MOD-en

```bash
python train.py dstc-en \
  --data-root data/DSTC10-MOD/en \
  --granite-text-model-path pretrained_weights/granite-embedding-311m-multilingual-r2 \
  --pvt-checkpoint pretrained_weights/PVT_v2_b2.pth
```

## 许可证

源代码采用 MIT 许可证发布。MultiChat、DSTC10-MOD、Granite 和 PVT 权重遵循各自的许可证。
