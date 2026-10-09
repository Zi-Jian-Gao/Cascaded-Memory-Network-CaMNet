#  ------------------------------------------------------------------------------------------
#  adapted from code with the following copyright:
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn

from typing import Dict

from .layers import LoRALayer

import torch.nn.init as init
import math

#module_list和nn.ParameterList的方式是否不一样？如果不一样就要重开一个utils，不然公用

def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'lora_' not in n and 'adapter' not in n:
            p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and \
                hasattr(m, 'bias') and \
                m.bias is not None:
                    m.bias.requires_grad = True
    else:
        raise NotImplementedError
    # Optimizer creation

def update_ema_epoch_lora(model: nn.Module,alpha,task_id,update_both, bias: str = 'none') -> None:
    lora0_params = {}
    # 第一遍遍历：收集lora0的参数
    for name, param in model.named_parameters():
        if 'lora_A.0' in name:
            # 以新的键存储参数，将'.0'替换为'.1'以便后续匹配lora1
            lora0_params[name.replace('lora_A.0', 'lora_A.1')] = param.clone()
        if 'lora_B.0' in name:
            # 以新的键存储参数，将'.0'替换为'.1'以便后续匹配lora1
            lora0_params[name.replace('lora_B.0', 'lora_B.1')] = param.clone()

        if 'adapter.0' in name:
            # 以新的键存储参数，将'.0'替换为'.1'以便后续匹配lora1
            lora0_params[name.replace('adapter.0', 'adapter.1')] = param.clone()

    # 第二遍遍历：将lora0的参数复制到lora1
    for name, param in model.named_parameters():
        if name in lora0_params:
            # 将lora0的参数直接复制给lora1
            if task_id == 0:
                param.data = lora0_params[name].data
            else:
                param.data = (1-alpha) * lora0_params[name].data + alpha * param.data


def lora_initial_lora(model: nn.Module, bias: str = 'none') -> None:
    lora0_params = {}
    # 第一遍遍历：收集lora1的参数
    for name, param in model.named_parameters():
        if 'lora_A.0' in name:
            # 以新的键存储参数，将'.1'替换为'.0'以便后续匹配lora0
            lora0_params[name.replace('lora_A.0', 'lora_A.1')] = param.clone()
        if 'lora_B.0' in name:
            # 以新的键存储参数，将'.1'替换为'.0'以便后续匹配lora0
            lora0_params[name.replace('lora_B.0', 'lora_B.1')] = param.clone()
        if 'adapter.0' in name:
            # 以新的键存储参数，将'.1'替换为'.0'以便后续匹配lora0
            lora0_params[name.replace('adapter.0', 'adapter.1')] = param.clone()

    # 第二遍遍历：将lora1的参数复制到lora0
    for name, param in model.named_parameters():
        if name in lora0_params:
            # 将lora0的参数直接复制给lora1
            param.data = lora0_params[name].data

def update_quick_lora_ini_slow(model: nn.Module,task_id) -> None:
    past_ema_lora_id = task_id
    ema_lora_id = task_id + 1

    for name, param in model.named_parameters():
        if f'lora_A.{past_ema_lora_id}' in name:
            # 使用Kaiming/He均匀初始化来初始化lora_A.0
            init.kaiming_uniform_(param, a=math.sqrt(5))
        if f'lora_B.{past_ema_lora_id}' in name:
            # 为lora_B.0初始化为零
            init.zeros_(param)
        if f'lora_A.{ema_lora_id}' in name:
            # 使用Kaiming/He均匀初始化来初始化lora_A.0
            init.kaiming_uniform_(param, a=math.sqrt(5))
        if f'lora_B.{ema_lora_id}' in name:
            # 为lora_B.0初始化为零
            init.zeros_(param)

        if f'adapter.{past_ema_lora_id}' in name:
            if 'down_proj.weight'in name:
                init.normal_(param,mean=0.0, std=0.02)
            if 'down_proj.bias' or 'up_proj.bias' or 'up_proj.weight' in name:
                init.zeros_(param)

        if f'adapter.{ema_lora_id}' in name:
            if 'down_proj.weight' in name:
                init.normal_(param, mean=0.0, std=0.02)
            if 'down_proj.bias' or 'up_proj.bias' or 'up_proj.weight'  in name:
                init.zeros_(param)

def update_ema_epoch_lora_slow(model: nn.Module,alpha,task_id,epoch) -> None:
    fast_lora_id = task_id
    slow_lora_id = task_id + 1

    lora0_params = {}
    # 第一遍遍历：收集lora0的参数
    for name, param in model.named_parameters():
        if f'lora_A.{fast_lora_id}' in name:
            # 以新的键存储参数，将'.0'替换为'.1'以便后续匹配lora1
            lora0_params[name.replace(f'lora_A.{fast_lora_id}', f'lora_A.{slow_lora_id}')] = param.clone()
        if f'lora_B.{fast_lora_id}' in name:
            # 以新的键存储参数，将'.0'替换为'.1'以便后续匹配lora1
            lora0_params[name.replace(f'lora_B.{fast_lora_id}', f'lora_B.{slow_lora_id}')] = param.clone()

        if f'adapter.{fast_lora_id}' in name:
            lora0_params[name.replace(f'adapter.{fast_lora_id}', f'adapter.{slow_lora_id}')] = param.clone()

    # 第二遍遍历：将lora0的参数复制到lora1
    for name, param in model.named_parameters():
        if name in lora0_params:
            if task_id == 0:
                param.data = lora0_params[name].data
            if epoch == 0:
                param.data = lora0_params[name].data
            else:
                param.data = (1-alpha) * lora0_params[name].data + alpha * param.data

def update_ema_epoch_lora_cls(model: nn.Module,alpha,task_id, bias: str = 'none') -> None:
    lora0_params = {}
    lora1_params = {}
    # 第一遍遍历：收集lora0的参数
    for name, param in model.named_parameters():
        if 'lora_A.0' in name:
            # 以新的键存储参数，将'.0'替换为'.1'以便后续匹配lora1
            lora0_params[name.replace('lora_A.0', 'lora_A.1')] = param.clone() #slow
            lora1_params[name.replace('lora_A.0', 'lora_A.2')] = param.clone() #fast
        if 'lora_B.0' in name:
            # 以新的键存储参数，将'.0'替换为'.1'以便后续匹配lora1
            lora0_params[name.replace('lora_B.0', 'lora_B.1')] = param.clone()
            lora1_params[name.replace('lora_B.0', 'lora_B.2')] = param.clone()
        if 'adapter.0' in name:
            # 以新的键存储参数，将'.0'替换为'.1'以便后续匹配lora1
            lora0_params[name.replace('adapter.0', 'adapter.1')] = param.clone()
            lora1_params[name.replace('adapter.0', 'adapter.2')] = param.clone()

    # 第二遍遍历：将lora0的参数复制到lora1
    for name, param in model.named_parameters():
        if name in lora0_params:
            # 将lora0的参数直接复制给lora1
            if task_id == 0:
                param.data = lora0_params[name].data
            else:
                param.data = (1-alpha) * lora0_params[name].data + alpha * param.data

        elif name in lora1_params:
            # 将lora0的参数直接复制给lora1
            if task_id == 0:
                param.data = lora1_params[name].data
            else:
                param.data = (1 - alpha + 0.1) * lora1_params[name].data + (alpha - 0.1) * param.data


def ini_lora_cls(model: nn.Module, bias: str = 'none') -> None:
    lora0_params = {}
    lora1_params = {}
    # 第一遍遍历：收集lora0的参数
    for name, param in model.named_parameters():
        if 'lora_A.0' in name:
            # 以新的键存储参数，将'.0'替换为'.1'以便后续匹配lora1
            lora0_params[name.replace('lora_A.0', 'lora_A.1')] = param.clone() #slow
            lora1_params[name.replace('lora_A.0', 'lora_A.2')] = param.clone() #fast
        if 'lora_B.0' in name:
            # 以新的键存储参数，将'.0'替换为'.1'以便后续匹配lora1
            lora0_params[name.replace('lora_B.0', 'lora_B.1')] = param.clone()
            lora1_params[name.replace('lora_B.0', 'lora_B.2')] = param.clone()
        if 'adapter.0' in name:
            # 以新的键存储参数，将'.0'替换为'.1'以便后续匹配lora1
            lora0_params[name.replace('adapter.0', 'adapter.1')] = param.clone()
            lora1_params[name.replace('adapter.0', 'adapter.2')] = param.clone()

    # 第二遍遍历：将lora0的参数复制到lora1
    for name, param in model.named_parameters():
        if name in lora0_params:
            # 将lora0的参数直接复制给lora1
            param.data = lora0_params[name].data
        elif name in lora1_params:
            # 将lora0的参数直接复制给lora1
            param.data = lora1_params[name].data

def lora_initial_ema(model: nn.Module, bias: str = 'none') -> None:
    lora1_params = {}
    # 第一遍遍历：收集lora1的参数
    for name, param in model.named_parameters():
        if 'lora_A.1' in name:
            # 以新的键存储参数，将'.1'替换为'.0'以便后续匹配lora0
            lora1_params[name.replace('lora_A.1', 'lora_A.0')] = param.clone()
        if 'lora_B.1' in name:
            # 以新的键存储参数，将'.1'替换为'.0'以便后续匹配lora0
            lora1_params[name.replace('lora_B.1', 'lora_B.0')] = param.clone()

        if 'adapter.1' in name:
            # 以新的键存储参数，将'.1'替换为'.0'以便后续匹配lora0
            lora1_params[name.replace('adapter.1', 'adapter.0')] = param.clone()

    # 第二遍遍历：将lora1的参数复制到lora0
    for name, param in model.named_parameters():
        if name in lora1_params:
            # 将lora0的参数直接复制给lora1
            param.data = lora1_params[name].data


