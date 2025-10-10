Fine-grained visual recognition (FGVR) requires not only high accuracy but also
human-aligned interpretability, particularly in safety-critical applications. While
human cognition naturally follows a coarse-to-fine reasoning process—rapid
holistic categorization for coarse-grained class followed by attention to local details for fine-grained class—existing post-hoc and ante-hoc interpretability methods fall short in capturing this hierarchy automatically. To address this gap, we
propose Bi-HiR, a novel Bidirectional Hierarchical Reasoning framework that
emulates human-like cognition by integrating top-down semantic reasoning with
bottom-up prototype-based explanations. Specifically, Bi-HiR: (1) leverages large
language model (LLM)-derived semantic priors to construct coarse-to-fine hierarchies without manual annotations; (2) introduces a joint optimization strategy
where top-down priors guide bottom-up prototype learning across semantic levels; and (3) produces interpretable, step-wise visual and semantic explanations.
Experiments on six FGVR benchmarks demonstrate that Bi-HiR achieves competitive SOTA performance and exhibits superior zero-shot generalization. The
results also reveal the superiority of Bi-HiR’s interpretability on human trust and
model error diagnose.

<img width="2060" height="1049" alt="Figure1_4" src="https://github.com/user-attachments/assets/ce9ec347-c64d-4ca3-9d15-3ebdda72e6ab" />



Prerequisites: PyTorch, NumPy, cv2, Augmentor

Recommended hardware: 2 NVIDIA GeForce RTX 3090 GPUs
