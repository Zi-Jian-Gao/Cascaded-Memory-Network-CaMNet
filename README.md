# CaMNet: Neuro-inspired cascaded memory coordination enables continual generalization

This repository contains the official implementation of **CaMNet**. To ensure reproducibility across different research domains, the codebase is partitioned based on specific experimental environments and model architectures.

## Repository Structure

The project is organized into two main directories:

* **`CamNet_ContinualMultiModal/`**: This directory contains the implementation for **Continual Multi-Modal Learning** tasks (e.g., Continual VQA). It is optimized for Vision-Language Models (VLMs) and focuses on representation memory and modality alignment.
* **`CM_ContinualRL/`**: This directory contains the implementation for **Continual Reinforcement Learning (CRL)**. It is tailored for decision-making agents and interaction with dynamic environments.

---

## Why the Separation?

We have decoupled the code into these two folders for the following reasons:

1.  **Environment Dependencies**: Continual RL often requires specific physics engines or simulators (e.g., Meta-World) and reinforcement learning frameworks that have different dependency versions compared to the large-scale Vision-Language processing libraries (e.g., HuggingFace) used in the Multi-Modal setup.
2.  **Model Architectures**: While the core "cascaded memory coordination" mechanism remains consistent, the backbone architectures differ significantly—ranging from high-capacity transformer-based encoders in the multi-modal tasks to MLP based networks in the RL tasks.
3.  **Data Pipelines**: The data loading, preprocessing, and evaluation metrics (e.g., Accuracy vs. Cumulative Reward) are domain-specific, and separation ensures a cleaner and more maintainable codebase.

## Getting Started

Please refer to the `README.md` file within each respective folder for detailed instructions on:
* Environment setup and `requirements.txt`.
* Data preparation and downloading.
* Training and evaluation scripts.