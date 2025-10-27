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

<figure style="text-align: center; margin-bottom: 2em;">
  <img width="2060" height="1049" alt="Figure1_4" src="https://github.com/user-attachments/assets/ce9ec347-c64d-4ca3-9d15-3ebdda72e6ab" />
  <figcaption>Figure 1. Four images from different categories are concatenated into a 2×2 grid as input. The results demonstrate that:
(1) Bi-HiR can effectively discriminate between fine-grained classes and provides trustworthy coarse-to-fine hierarchical explanations at both conceptual and visual levels;
(2) Other interpretable methods fail to make reliable decisions on the 2×2 grid input and are unable to generate trustworthy explanations;
(3) The visual explanations produced by Bi-HiR exhibit high consistency between single-image and 2×2 grid inputs, whereas the compared methods lack such consistency in their explanations.</figcaption>
</figure>


<figure style="text-align: center; margin-bottom: 2em;">
  <img width="1397" height="852" alt="method" src="https://github.com/user-attachments/assets/47546921-e35a-448b-93a5-2b8f8b22530f" />
  <figcaption>Figure 2. The proposed architecture comprises five key components:
(a) Post-hoc Enhancement,
(b) Multi-scale Aggregation,
(c) Semantic Hierarchy Construction,
(d) Bi-HiR Training, and
(e) Bi-HiR Inference.
Together, these modules facilitate a top-down, human-like reasoning process that progressively transitions from generalist to specialist understanding.
</figcaption>
</figure>

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
    --save_path ${SAVE_PATH} \
```

- Here the datatxt_train, datatxt_OFA1, datatxt_OFA2, datatxt_OFA3, datatxt_val are the path of data list which are provided in the above link.
- Train_num is the number of training process to ensure the stability of result.
- Patience is the parameter of earlystop strategy to stop training when accuracy of validation set does not improve.
- Arch is the type of backbone which can be selected during Densenet121_PIHA, Aconvnet_PIHA and MSNet_PIHA.
- Part_num is the numbers of clusters in data preparation and part_num of our data is 4.
- Attention_setting decide whether to use our PIHA.
