# Dissecting-VLMs

This repository builds upon the foundational work of the [LLaVA](https://github.com/haotian-liu/llava) project. Special thanks to [@haotian-liu](https://github.com/haotian-liu) for making the research on multimodal models more accessible through open-sourcing the `llava` repository.

### Enhancements in This Repository

- **Soft Mixture of Expert Projector Layer**: A new architectural component for improved model performance.
- **Integration of Advanced Models**:
  - SigLIP
  - AIMv2
  - internViT
  - DinoV2
- **Multinode Training with SLURM**: Includes SLURM scripts to facilitate multinode training and scaling.

### Code Modifications

Key changes were made to the following modules:
- `llava/model`: Updates to support new features and integrations.
- `llava/train`: Modifications for advanced training workflows.

### Getting Started

To use the features of this repository, clone it and explore the updated `model` and `train` directories. Detailed instructions for multinode training using SLURM are available in the provided scripts.

### Acknowledgments

This repository would not have been possible without the contributions of the LLaVA project and its authors. Their work lays the groundwork for further exploration and innovation in vision-language models.

For more details, refer to the [LLaVA repository](https://github.com/haotian-liu/llava). 
