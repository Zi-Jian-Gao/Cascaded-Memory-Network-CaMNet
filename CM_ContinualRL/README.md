# Neuro-inspired cascaded memory coordination enables continual generalization

---

This is the official PyTorch implementation of the paper **"Neuro-inspired cascaded memory coordination enables continual generalization"** (submitted to *Nature Machine Intelligence*).
This code implements the cascaded memory architecture for continual reinforcement learning experiments, allowing for the reproduction of the results presented in our [paper](xxx).

We built our code implementation based on the [garage](https://github.com/rlworkgroup/garage) and [Reset-Distill](https://github.com/hongjoon0805/Reset-Distill) repositories.

---

## Installation & Environments

Before running experiments, please set up the environment using the following commands:

```
conda env create --file environment.yaml
conda activate your_env_name
```

---

## Running experiments

The basic form of the execution command is as follows:

```
python3 main_garage.py --cl_method [cl_method] --long_task_seq [sequence of tasks] --seed [seed] --device_type [device num] --proc_name [your process name] --wandb [True or False] --use_exploration True
```

Available options:
- cl_method: `finetuning`, `IN`, `PNN`, `finetuning_CM`, `pandc`, `pandc_CM`, `bc`, `bc_CM`.
- long_task_seq: Difficulty of the task sequence (hard for main text experiments, easy for supplementary).
- seed: specific random seed.
- device_type: GPU device index.
- proc_name: Name of the process (for logging).
- wandb: Whether to use Weights & Biases logging.
---

### Continual Learning Experiments (Long Task Sequence)
Use the following commands to run experiments on the "hard" task sequence:

```
# Finetune
python3 main_garage.py --cl_method finetuning --long_task_seq hard --seed 0 --proc_name Finetune_Hard --device_type 0 --wandb False --use_exploration True

# Incremental Networks
python3 main_garage.py --cl_method IN --long_task_seq hard --seed 0 --proc_name IN_Hard --device_type 0 --wandb False --use_exploration True

# Progressive Neural Networks
python3 main_garage.py --cl_method PNN --long_task_seq hard --seed 0 --proc_name PNN_Hard --device_type 0 --wandb False --use_exploration True

# finetune+Cascaded Memory (Ours)
python3 main_garage.py --cl_method finetuning_CM --long_task_seq hard --seed 0 --proc_name finetune_CM_Hard --device_type 0 --wandb False --use_exploration True

# Progress&Compress
python3 main_garage.py --cl_method pandc  --long_task_seq hard --seed 0 --proc_name PC_Hard --device_type 0 --wandb False --use_exploration True

# Progress&Compress+Cascaded Memory (Ours)
python3 main_garage.py --cl_method pandc_CM --long_task_seq hard --seed 0 --proc_name PC_CM_Hard --device_type 0 --wandb False --use_exploration True

# Behavioural Cloning
python3 main_garage.py --cl_method bc --long_task_seq hard --seed 0 --proc_name BC_Hard --device_type 0 --wandb False --use_exploration True

# Behavioural Cloning+Cascaded Memory (Ours)
python3 main_garage.py --cl_method bc_CM --long_task_seq hard --seed 0 --proc_name BC_CM_Hard --device_type 0 --wandb False --use_exploration True

```
---

### Single task experiment

To run a single task experiment, use the command below:

```
python3 main_garage.py --cl_method finetuning --env_type metaworld --task_seq_idx 19 --seed 0 --proc_name Single_task_faucet-open
```
You can run experiments on other tasks (e.g., faucet-open, push) by changing the --task_seq_idx.
