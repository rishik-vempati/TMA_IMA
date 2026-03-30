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

### TMA : Text Matrix Adapter
