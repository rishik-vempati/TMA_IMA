# IMA & TMA: Efficient Test-Time Adaptation for VLMs via Linear Transformation in Embedding-Space

This repo provides the source code of our approach that was accepted in CVPR'26 ViScale workshop.

## Abstract
Large-scale Vision-Language Models (VLMs) have set new benchmarks in zero-shot learning; however, their performance remains brittle under distribution shifts at test time.
While existing Test-Time Adaptation (TTA) methods often rely on prompt tuning or input-space optimization, they incur significant computational overhead and scale poorly with class cardinality.
To bridge this gap, we propose two lightweight, sample-wise alignment strategies: Image Matrix Adapter (IMA) and Text Matrix Adapter (TMA).
Unlike previous methods, IMA and TMA apply linear corrections directly in the embedding space, thereby restoring cross-modal alignment with a single test sample.
This approach drastically reduces memory and computational requirements, as the adaptation cost remains independent of the number of target classes. 
Extensive evaluations across diverse out-of-distribution (OOD) benchmarks and cross-dataset scenarios demonstrate that our methods achieve competitive accuracy while being significantly more efficient than state-of-the-art prompt-based adaptation, making them ideal for resource-constrained deployment.

## Methodology
### IMA : Image Matrix Adapter
![img](images/methodology/IMA.png)

### TMA : Text Matrix Adapter
![img](images/methodology/TMA.png)

## Prerequisites
### Hardware
Experiments were conducted on a remote GPU server with the following configuration:

- GPUs: 8 × NVIDIA Tesla V100-SXM2 (32 GB VRAM each)
- CUDA Version: 12.2
- Platform: Linux-based system

#### Resource Usage
- GPU memory used per experiment: ~1.5–2 GB 
- Adaptation-specific memory usage: ~0.72 GB

### Software Environment
Experiments were conducted using the following software setup:

- OS: Linux (Ubuntu-based)
- Python: 3.7.12
