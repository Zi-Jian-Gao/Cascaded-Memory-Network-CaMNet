#  ------------------------------------------------------------------------------------------
#  adapted from code with the following copyright:
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import math
from typing import Optional, List

class Base_adapter(nn.Module):
    def __init__(self, dim, r=16, activation=nn.GELU, drop=0.):
        super().__init__()
        self.down_proj = nn.Linear(dim, r)  # Down-projection (bottleneck)
        self.act = activation()  # Activation function
        self.up_proj = nn.Linear(r, dim)  # Up-projection
        self._init_weights()
        if drop > 0.:
            self.drop = nn.Dropout(p=drop)
        else:
            self.drop = lambda x: x

    def _init_weights(self):
        # with torch.no_grad():
        self.down_proj.weight.data.normal_(mean=0.0, std=0.02)
        self.down_proj.bias.data.zero_()
        self.up_proj.weight.data.normal_(mean=0.0, std=0.02)
        self.up_proj.bias.data.zero_()

    def forward(self, x):
        x = self.down_proj(x)
        x = self.act(x)
        x = self.up_proj(x)
        x = self.drop(x)
        return x

class Adapter(nn.Module):
    def __init__(self, dim, r=16, activation=nn.GELU, drop=0., agent=None):
        super().__init__()
        self.r = r
        self.activation = activation
        self.drop = drop
        self.agent = agent
        numA = self.get_num_adapters()

        if numA==1:
            self.adapter = nn.ModuleList([Base_adapter(dim, self.r, self.activation, self.drop)])
        else:
            self.adapter = nn.ModuleList([Base_adapter(dim, self.r, self.activation, self.drop) for _ in range(numA)])
        self.apply_lock_policy()

    def get_num_adapters(self):
        numA = 1
        if (self.agent is not None) and self.agent.multi:
            if self.agent.slow:
                if self.agent.task_id == 0:
                    return numA
                numA = self.agent.get_num_tasks() + 1
            else:
                numA = self.agent.get_num_tasks()
        elif (self.agent is not None) and self.agent.ema:
            numA = self.agent.get_num_tasks()
        elif (self.agent is not None) and self.agent.LwF:
            numA = self.agent.get_num_tasks()
        elif (self.agent is not None) and self.agent.cls:
            numA = self.agent.get_num_tasks()
        return numA

    def apply_lock_policy(self):
        #在这里要修改一下，ema的情况下multi-lora一直训练的是第一个所以应该训练
        numA = self.get_num_adapters()
        if self.agent.ema or self.agent.LwF:
            for param in self.adapter[0].parameters():
                param.requires_grad = True
            for param in self.adapter[1].parameters():
                param.requires_grad = False
        elif self.agent.cls:
            for param in self.adapter[0].parameters():
                param.requires_grad = True
            for param in self.adapter[1].parameters():
                param.requires_grad = False
            for param in self.adapter[2].parameters():
                param.requires_grad = False
        else:
            if numA > 1:
                if self.agent.slow:
                    for i in range(numA - 2):
                        for param in self.adapter[i].parameters():
                            param.requires_grad = False
                    for param in self.adapter[-1].parameters():
                        param.requires_grad = False
                    for param in self.adapter[-2].parameters():
                        param.requires_grad = True
                else:
                    for i in range(numA - 1):
                        for param in self.adapter[i].parameters():
                            param.requires_grad = False
            else:
                for param in self.adapter[0].parameters():
                    param.requires_grad = True


    def forward(self, x: torch.Tensor, wild_id = None):
        result = torch.zeros_like(x)
        num = self.get_num_adapters()
        if isinstance(self.adapter, nn.ModuleList) and num != 1:
            if self.agent.ema or self.agent.LwF:
                if self.agent.fuse_type in ['ema'] or self.agent.fuse_type in ['LwF']:
                    result += self.adapter[1](x)
                else:
                    result += self.adapter[0](x)
            elif self.agent.cls:
                if self.agent.fuse_type in ['slow']:
                    # EMA，在ema推理的时候只看第二个ema_lora的
                    result += self.adapter[1](x)
                elif self.agent.fuse_type in ['fast']:
                    # EMA，在ema推理的时候只看第二个ema_lora的
                    result += self.adapter[2](x)
                else:
                    # 正常ema训练的时候看第一个lora的
                    result += self.adapter[0](x)
            else:
                if self.agent.fuse_type == 'multi-zsl-single' and wild_id is not None:
                    numA = wild_id[0]
                    if self.agent.slow and numA == -1:
                        numA = self.get_num_adapters() - 2
                        for i in range(numA):
                            result += self.adapter[i](x)
                        result += self.adapter[-1](x)
                    else:
                        for i in range(numA):
                            result += self.adapter[i](x)
                else:
                    numA = self.get_num_adapters()
                    if not self.agent.eval:
                        if self.agent.slow:
                            numA = self.get_num_adapters() - 1
                        for i in range(numA):
                            result += self.adapter[i](x)
                    else:
                        numA = self.get_num_adapters() - 2
                        for i in range(numA):
                            result += self.adapter[i](x)
                        result += self.adapter[-1](x)
        else:
            result += self.adapter[0](x)
        return x + result