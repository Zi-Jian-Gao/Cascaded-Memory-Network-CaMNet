"""PyTorch algorithms."""
# isort:skip_file

# SAC needs to be imported before MTSAC
from garage.torch.algos.sac import SAC
from garage.torch.algos.mtsac import MTSAC
from garage.torch.algos.vpg import VPG
from garage.torch.algos.ppo import PPO
from garage.torch.algos.bc import BC_SAC, BC_PPO
from garage.torch.algos.bc_ours import BC_SAC_ours, BC_PPO_ours
from garage.torch.algos.rnd import RND_SAC, RND_PPO
from garage.torch.algos.finetuning import Finetuning_SAC, Finetuning_PPO
from garage.torch.algos.finetuning_ours import Finetuning_SAC_ours, Finetuning_PPO_ours

from garage.torch.algos.ewc import EWC_SAC, EWC_PPO
from garage.torch.algos.pandc import P_and_C_SAC, P_and_C_PPO
from garage.torch.algos.pandc_ours import P_and_C_SAC_ours, P_and_C_PPO_ours
__all__ = [
    'SAC', 'MTSAC', 'PPO', 'VPG',
    'BC_SAC', 'BC_PPO',
    'BC_SAC_ours', 'BC_PPO_ours',
    'RND_SAC', 'RND_PPO',
    'Finetuning_SAC', 'Finetuning_PPO',
    'Finetuning_SAC_ours', 'Finetuning_PPO_ours',
    'EWC_SAC', 'EWC_PPO',
    'P_and_C_SAC', 'P_and_C_PPO',
    'P_and_C_SAC_ours', 'P_and_C_PPO_ours'
]
