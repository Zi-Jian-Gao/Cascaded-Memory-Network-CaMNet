import numpy as np
import torch


from garage.torch.algos import MTSAC, PPO
from garage.torch import as_torch_dict

import wandb

class Finetuning_SAC_ours(MTSAC):

    def __init__(self, **sac_kwargs):
        super().__init__(**sac_kwargs)

    def on_task_start(self, seq_idx):
        # print(seq_idx)

        self.policy._module._shared_mean_log_std_network.expand_trunk()

        new_params = [
            p for p in self.policy.parameters()
            if p.requires_grad and id(p) not in self._policy_params_ids
        ]
        if new_params:
            self._policy_optimizer.add_param_group({'params': new_params})

        self._policy_params_ids.update(id(p) for p in new_params)



class Finetuning_PPO_ours(PPO):

    def __init__(self, **ppo_kwargs):
        super().__init__(**ppo_kwargs)


    def on_task_start(self, seq_idx):

        self.policy._module._shared_mean_log_std_network.expand_trunk()

        new_params = [
            p for p in self.policy.parameters()
            if p.requires_grad and id(p) not in self._policy_params_ids
        ]
        if new_params:
            self._policy_optimizer.add_param_group({'params': new_params})
        self._policy_params_ids.update(id(p) for p in new_params)
