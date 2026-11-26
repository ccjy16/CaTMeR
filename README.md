# CaTMeR

[简体中文](README.zh-CN.md)

Training code for **CaTMeR: Emotion- and Intent-Aware Sticker Retrieval with Candidate-Conditioned Temporal Memory**. CaTMeR extends the implementation of **MGTM: Intent- and Emotion-Guided Temporal Memory for Cross-Session Sticker Retrieval**, accepted at CEI 2026. [CEI 2026 implementation](https://github.com/ccjy16/MGTM)

CaTMeR stands for Candidate-Conditioned Temporal Memory for Retrieval. It selects earlier texts from the conversation, encodes their relative order, and lets each candidate sticker attend to the selected entries to refine its score. Granite Embedding 311M Multilingual R2 encodes the text, and PVT v2 B2 extracts global and local visual features. MultiChat uses intent supervision, and both language versions of DSTC10-MOD use emotion supervision.

## Base weights

Download the five Granite files from the pinned revision and the official ImageNet pretrained PVT v2 B2 checkpoint. Use the layout below with the training commands in this README. [Granite Embedding 311M Multilingual R2](https://huggingface.co/ibm-granite/granite-embedding-311m-multilingual-r2/tree/44399559930365213510b1ee2eb15ded83374f0e), [PVT v2 classification weights](https://github.com/whai362/PVT/tree/v2/classification)

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

## Data layout

Obtain each benchmark under its own terms. The commands below use a repository-relative `data/` directory. Pass the actual paths when storing the data elsewhere.

### MultiChat

Use the preprocessed `all` configuration, which combines MultiChat's five modes. Each `pairs/` or `*_imgs/` directory is an LMDB store. Each pair record retains the mode it came from.

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

Prepare the Chinese and English versions separately. Each language directory has the structure shown below. `sticker_emotion_top2_train.json` maps sticker IDs to up to two emotion labels derived from that language's training split.

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
    `-- same layout as zh/
```

## Training

Run the commands from the repository root. By default, training uses the first visible GPU. Set `CUDA_VISIBLE_DEVICES` or pass `--gpu-id` to select a device. `--gpu-id` is relative to the list of visible devices.

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

## License

The source code is released under the MIT License. MultiChat, DSTC10-MOD, Granite, and PVT weights remain subject to their respective licenses.
