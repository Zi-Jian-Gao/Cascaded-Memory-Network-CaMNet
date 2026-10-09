source ~/.bashrc
conda activate RL

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
