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
# Base Adapter class #
#####################
#fully-fintuning
class CFT(Base):
    def __init__(self, agent_config):
        super(CFT, self).__init__(agent_config)
        self.adapter = False
        self.r = int(self.mu)
        # initialize prompts
        self.freeze_encoders = False
        self.freeze_heads = True
        self.layer_keys = [0]
#adapter
class Adapter(Base):
    def __init__(self, agent_config):
        super(Adapter, self).__init__(agent_config)
        self.adapter = True
        self.r = int(self.mu)

        # initialize prompts
        self.freeze_encoders = True
        self.freeze_heads = True
        self.layer_keys = [0]

#layred-adapter
class MultiAdapter(Adapter):
    def __init__(self, agent_config):
        super(MultiAdapter, self).__init__(agent_config)
        self.freeze_heads = True
        self.multi = True
        # 推理时看前面所有Adapter的结果
        self.fuse_type = 'last'


#ConStruct-VL: Data-Free Continual Structured VL Concepts Learning
class AdvTextMultiAdapter(MultiAdapter):
    def __init__(self, agent_config):
        super(AdvTextMultiAdapter, self).__init__(agent_config)
        self.train_distill_type = 'adv_text'
        #推理时看前面所有Adapter的结果
        self.fuse_type = 'last'

#EMA
class EMAAdapter(Adapter):
    def __init__(self, agent_config):
        super(EMAAdapter, self).__init__(agent_config)
        self.freeze_heads = True
        self.ema = True

class LwF(Adapter):
    def __init__(self, agent_config):
        super(LwF, self).__init__(agent_config)
        self.freeze_heads = True
        self.LwF = True

class ER(Adapter):
    def __init__(self, agent_config):
        super(ER, self).__init__(agent_config)
        self.freeze_heads = True
        self.memory_size = 20


class EWC(Adapter):
    def __init__(self, agent_config):
        super(EWC, self).__init__(agent_config)
        self.freeze_heads = True
        self.ewc = True
        self.fisher = None
        self.mean = None

class GEM(Adapter):
    def __init__(self, agent_config):
        super(GEM, self).__init__(agent_config)
        self.freeze_heads = True
        self.gem = True
        self.memory_size = 20

#CLS
class CLS(Adapter):
    def __init__(self, agent_config):
        super(CLS, self).__init__(agent_config)
        self.freeze_heads = True
        self.cls = True
        self.memory_size = 20

#ZAF
class ZAF(Adapter):
    def __init__(self, agent_config):
        super(ZAF, self).__init__(agent_config)
        self.freeze_heads = True
        self.ema = True
        self.train_distill_type = 'ema-zsl-single'
        #the random setting shuffles the texts and makes them unpaired.
        self.random = True
        self.zaf = True

class MultiAdapter_wild(Adapter):
    def __init__(self, agent_config):
        super(MultiAdapter_wild, self).__init__(agent_config)
        self.freeze_heads = True
        self.ema = False
        self.multi = True
        self.train_distill_type = 'multi-zsl-single'
        self.random = False
        self.zaf = True


class Ours(Adapter):
    def __init__(self, agent_config):
        super(Ours, self).__init__(agent_config)
        self.freeze_heads = True
        self.ema = False
        self.multi = True
        self.train_distill_type = 'multi-zsl-single'
        self.random = False
        self.zaf = True
        self.slow = True
        self.eval = False

class Ours_LM(Adapter):
    def __init__(self, agent_config):
        super(Ours_LM, self).__init__(agent_config)
        self.freeze_heads = True
        self.ema = False
        self.multi = True
        self.train_distill_type = 'multi-zsl-single'
        self.random = False
        self.LM = True
        self.slow = True
        self.eval = False

