# Neuro-inspired cascaded memory coordination enables continual generalization

This repository contains the official PyTorch implementation of the paper **"Neuro-inspired cascaded memory coordination enables continual generalization"** (submitted to *Nature Machine Intelligence*).

This code implements the **Cascaded Memory Network (CaMNet)** for continual vision-language learning. It supports the reproduction of main experiments presented in the paper, including benchmarks on VQAv2, GQA, PathVQA, SVLC, and Embodied VQA tasks.

**Acknowledgements:** This codebase is built upon [ZAF](https://github.com/Zi-Jian-Gao/Stabilizing-Zero-Shot-Prediction-ZAF/), [ConStruct-VL](https://github.com/jamessealesmith/ConStruct-VL), and [BLIP](https://github.com/salesforce/BLIP).

---
## 🛠️ Installation

Before running experiments, please set up the environment using the following commands:

```
conda env create --file environment.yaml
conda activate your_env_name
pip install flash-attn==2.7.0.post2 --no-build-isolation
```

---
## ⚙️ Configuration & Setup
**Dataset Preparation:**
Please configure the dataset paths in the configuration files located at `configs/task_config/*.yaml` to match your local setup. The required datasets can be downloaded from the sources below:

* **VQA Task:**
    * **VQAv2:** [Train (2014)](http://images.cocodataset.org/zips/train2014.zip) | [Val (2014)](http://images.cocodataset.org/zips/val2014.zip) | [Test (2015)](http://images.cocodataset.org/zips/test2015.zip)
    * **GQA:** [Images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
    * **PathVQA:** [HuggingFace Dataset](https://huggingface.co/datasets/flaviagiammarino/path-vqa)
    * **Our Embodied Data:** [Download](https://drive.google.com/drive/folders/1yb2BGHc7zibLEEkg6_WYUNGtRrU3QBCh?usp=drive_link)
    * IMPORTANT: To ensure the dataloader functions correctly, the Embodied Data directory must be placed within the same parent folder as your VQAv2 images (e.g., both should reside in your data/coco/ or equivalent path).

* **SVLC Task:**
    * Based on protocols from VL-Checklist, utilizing two popular publicly available VL datasets **Visual Genome (VG)** and **Visual Attributes in the Wild (VAW)**.
    * **Download:** [VG Images (Part 1)](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) | [VAW Images (Part 2)](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)

**Pre-trained Model Preparation:**
Update the `pretrained` and `LM_path` arguments in your benchmark configuration files (located at `configs/task_config/*.yaml`) to point to your local checkpoints.

* **Vision-Language Backbone:**
    * Download the BLIP checkpoint from [here](https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base.pth).
    * Set this path to `pretrained`.

* **Language Model:**
    * Download **Qwen2.5-3B-Instruct** from [HuggingFace](https://hf-mirror.com/Qwen/Qwen2.5-3B-Instruct/tree/main).
    * Set this path to `LM_path`.

---
## 🚀 Running Experiments

### 1. Continual Learning Benchmarks

The general command to run continual learning experiments is:
```
torchrun --nproc_per_node=8 run_me.py \
    --config [config_path] \
    --agent_name [method_name] \
    --external_lr [learning_rate] \
    --loss_alpha [lambda_CR] \
    --ema_alpha [consolidation_factor]
    --seed [seed] \
    --output_dir [save_directory]
```
**Arguments:**

* `--config`: Path to the benchmark configuration file.
    * Options: `./configs/vqa/vqa.yaml`, `./configs/gqa/gqa.yaml`, `./configs/pathvqa/pathvqa.yaml`, `./configs/svlc/svlc.yaml`.
* `--agent_name`: The continual learning method to evaluate.
    * Baselines: `ER`, `EWC`, `GEM`, `LwF`, `LwF_ER`, `LAE`, `LAE_ER`, `CLS`, `CLS_ER`.
    * **Ours:**  `LwF_CM`, `CM_ER`, `LwF_CR`, `LAE_CR`, `CLS_CR`, `CM_CR`, `CaMNet`.
* `--external_lr`: Learning rate.
    * `0.00125` for VQAv2, GQA, SVLC; `0.002` for PathVQA.
* `--loss_alpha`: Regularization weight (or reactivation weight).
    * `1.0` for VQAv2, PathVQA, SVLC; `0.1` for GQA.
* `--ema_alpha`: Consolidation Factor (defaulted as 0.85).
* `--seed`: Random seed for reproducibility.
* `--output_dir`: Directory to save logs and checkpoints.

### 2. Single-Task Baseline

To run a single-task experiment (e.g., to calculate Proactive Interference or joint training bounds), point `--config` to a specific task configuration file and set the agent to `CLA`.

*Example:*
```bash
torchrun --nproc_per_node=8 run_me.py --config ./configs/vqa/color.yaml --agent_name CLA ...
```

### 3. Embodied VQA Experiment
To adapt a model pre-trained on VQAv2 to the Embodied VQA setting:
```bash
torchrun --nproc_per_node=8 run_embodied_vqa.py \
    --config ./configs/embodied_vqa/Embodied_vqa.yaml \
    --agent_name [method_name] \
    --output_dir [path_to_vqav2_pretrained_model_dir] \
    --external_lr 0.00125 \
    --loss_alpha 1.0 \
    --ema_alpha 0.2
```
Note: Ensure `output_dir` points to the directory containing the checkpoint trained on the standard VQAv2 benchmark. The script will load the pre-trained weights from this directory and continue training on the embodied sequence.

---
## 📊 Results and Metrics
Upon completion, checkpoints and results are saved in the `output_dir/task_models/` and `output_dir/final_result/` directories, respectively.

* **`cl_matrix.yaml`**: This file records the test accuracy of every learned task after each training stage.
    * **Final Accuracy (FA):** Calculated as the **average** of the accuracies in the final row of the matrix (i.e., the performance of all tasks after the last training stage).
    * **Retroactive Interference (RI):** Calculated as the **average difference** between each task's final accuracy and its accuracy immediately after it was first learned.
    * **Proactive Interference (PI):** Calculated as the **average difference** between each task's "first-learned" accuracy in the continual learning sequence and its baseline accuracy from the **Single-Task** experiment.
    * You can get the FA and RI by running the command ```python calculate.py --log cl_matrix.yaml```.

