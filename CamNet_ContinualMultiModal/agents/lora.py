import torch
import torch.nn as nn
from torch.nn import functional as F
import copy
import numpy as np
from pathlib import Path
import shutil
import os
import utils
import torch.distributed as dist
from agents.base import Base

#####################
#fully-fintuning
#####################
class CFT(Base):
    def __init__(self, agent_config):
        super(CFT, self).__init__(agent_config)
        self.lora = False
        self.r = int(self.mu)
        # initialize prompts
        self.freeze_encoders = False
        self.freeze_heads = True
        self.layer_keys = [0]
        self.CFT = True

#####################
# Base LoRa class #
#####################

class LoRa(Base):
    def __init__(self, agent_config):
        super(LoRa, self).__init__(agent_config)
        self.lora = True
        self.r = int(self.mu)
        # initialize prompts
        self.freeze_encoders = True
        self.freeze_heads = True
        self.layer_keys = [0]

class CLA(Base):
    def __init__(self, agent_config):
        super(CLA, self).__init__(agent_config)
        self.lora = True
        self.r = int(self.mu)
        # initialize prompts
        self.freeze_encoders = True
        self.freeze_heads = True
        self.layer_keys = [0]

class ER(LoRa):
    def __init__(self, agent_config):
        super(ER, self).__init__(agent_config)
        self.freeze_heads = True
        self.memory_size = 10


class LwF(LoRa):
    def __init__(self, agent_config):
        super(LwF, self).__init__(agent_config)
        self.freeze_heads = True
        self.LwF = True

class LwF_ER(LoRa):
    def __init__(self, agent_config):
        super(LwF_ER, self).__init__(agent_config)
        self.freeze_heads = True
        self.LwF = True
        self.LwF_ER = True
        self.memory_size = 10 #per task

class LwF_CR(LoRa):
    def __init__(self, agent_config):
        super(LwF_CR, self).__init__(agent_config)
        self.freeze_heads = True
        self.LwF = True
        self.LM = True

class LwF_CM(LoRa):
    def __init__(self, agent_config):
        super(LwF_CM, self).__init__(agent_config)
        self.freeze_heads = True
        self.multi = True
        self.lwf_ours = True
        self.train_distill_type = 'multi-zsl-single'

class LAE(LoRa):
    def __init__(self, agent_config):
        super(LAE, self).__init__(agent_config)
        self.freeze_heads = True
        self.ema = True
        self.cur_distill = True

class LAE_ER(LoRa):
    def __init__(self, agent_config):
        super(LAE_ER, self).__init__(agent_config)
        self.freeze_heads = True
        self.ema = True
        self.memory_size = 10

class LAE_CR(LoRa):
    def __init__(self, agent_config):
        super(LAE_CR, self).__init__(agent_config)
        self.freeze_heads = True
        self.ema = True
        self.train_distill_type = 'ema-zsl-single'
        self.LM = True

class CLS(LoRa):
    def __init__(self, agent_config):
        super(CLS, self).__init__(agent_config)
        self.freeze_heads = True
        self.cls = True
        self.cur_distill = True

class CLS_ER(LoRa):
    def __init__(self, agent_config):
        super(CLS_ER, self).__init__(agent_config)
        self.freeze_heads = True
        self.cls = True
        self.memory_size = 10

class CLS_CR(LoRa):
    def __init__(self, agent_config):
        super(CLS_CR, self).__init__(agent_config)
        self.freeze_heads = True
        self.cls = True
        self.LM = True

class CM(LoRa):
    def __init__(self, agent_config):
        super(CM, self).__init__(agent_config)
        self.freeze_heads = True
        self.LwF = True

class CM_ER(LoRa): #i.e., ER-CM
    def __init__(self, agent_config):
        super(CM_ER, self).__init__(agent_config)
        self.freeze_heads = True
        self.multi = True
        self.ours_id = True
        self.train_distill_type = 'multi-zsl-single'
        self.memory_size = 10

class CM_CR(LoRa):
    def __init__(self, agent_config):
        super(CM_CR, self).__init__(agent_config)
        self.freeze_heads = True
        self.multi = True
        self.train_distill_type = 'multi-zsl-single'
        self.LM = True

#ZAF
class ZAF(LoRa):
    def __init__(self, agent_config):
        super(ZAF, self).__init__(agent_config)
        self.freeze_heads = True
        self.ema = True
        self.train_distill_type = 'ema-zsl-single'
        #the random setting shuffles the texts and makes them unpaired.
        self.zaf = True

class ZAF_CM(LoRa):
    def __init__(self, agent_config):
        super(ZAF_CM, self).__init__(agent_config)
        self.freeze_heads = True
        self.multi = True
        self.train_distill_type = 'multi-zsl-single'
        self.zaf = True
        self.zaf_ours = True

class EWC(LoRa):
    def __init__(self, agent_config):
        super(EWC, self).__init__(agent_config)
        self.freeze_heads = True
        self.ewc = True
        self.fisher = None
        self.mean = None

class GEM(LoRa):
    def __init__(self, agent_config):
        super(GEM, self).__init__(agent_config)
        self.freeze_heads = True
        self.gem = True
        self.memory_size = 10

class CaMNet(LoRa):
    def __init__(self, agent_config):
        super(CaMNet, self).__init__(agent_config)
        self.freeze_heads = True
        self.multi = True
        self.train_distill_type = 'multi-zsl-single'
        self.LM = True
        self.slow = True
