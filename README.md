# Bidirectional Hierarchical Reasoning for Fine-grained Visual Recognition

---

## 0. Table of Contents

- [Introduction](#1-introduction)
- [Features](#2-features)
- [Contributions](#3-contributions)
- [Getting Started](#4-getting-started)

---

## 1. Introduction

This is the official implementation for paper "Bidirectional Hierarchical Reasoning for Fine-grained Visual Recognition".

## 2. Features

<img width="2060" height="1049" alt="Figure1_4" src="https://github.com/user-attachments/assets/ce9ec347-c64d-4ca3-9d15-3ebdda72e6ab" />

## 3. Contributions

- We propose a novel Bidirectional Hierarchical Reasoning framework for FGVR to emulate human top-down and bottom-up cognition mechanism.

- We construct LLM-derived top-down semantic priors and propose a Bi-HiR optimization process, that achieves hierarchical interpretability in both semantic reasoning and visual explanations.

- Extensive evaluations demonstrate Bi-HiR achieves competitive SOTA performance and significant improvements zero-shot generalization, human trust, and model diagnose.

## 4. Getting Started

### 4.1 Data Preparation

### 4.2 Training
To train Bi-HiR, run the following command

```bash
python main.py \
    --dataset ${DATASET} \
    --arch 'ResNet-50_Bi-HiR' \
    --proto_num '10' \
    --target_num  '3'\
    --epochs 120 \
    --batch_size 32 \
    --device '0' \
    --arch 'Densenet121_PIHA' \
    --attention_setting True \
    --save_path ${SAVE_PATH} \
```

- Here the datatxt_train, datatxt_OFA1, datatxt_OFA2, datatxt_OFA3, datatxt_val are the path of data list which are provided in the above link.
- Train_num is the number of training process to ensure the stability of result.
- Patience is the parameter of earlystop strategy to stop training when accuracy of validation set does not improve.
- Arch is the type of backbone which can be selected during Densenet121_PIHA, Aconvnet_PIHA and MSNet_PIHA.
- Part_num is the numbers of clusters in data preparation and part_num of our data is 4.
- Attention_setting decide whether to use our PIHA.
