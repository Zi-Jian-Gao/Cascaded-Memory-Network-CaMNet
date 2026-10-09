'''
 adapted from code with the following copyright:
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''
import argparse
import os
import copy
import yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import json
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from models.blip_nlvr import blip_nlvr
from quadprog import solve_qp
import utils
from task_trainers.nlvr_list import extract_nlvr_concept
from utils import cosine_lr_schedule, warmup_lr_schedule, count_parameters
from data import create_dataset, create_sampler, create_loader, create_zsl_dataset
from task_trainers.LM import Create_LM,generate_new_concept
import loralib as lora
from itertools import cycle

def train_ours(model, data_loader,language_model,tokenizer,optimizer, epoch, device, config, agent):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('our_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 10

    for i, batch_data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        #task data loss cal
        image0, pos, neg, idx = batch_data
        text = pos + neg
        image0 = image0.repeat(2, 1, 1, 1)
        targets = torch.zeros((len(text, )), dtype=torch.int64)
        targets[:len(pos)] = 1
        images = image0
        images, targets = images.to(device), targets.to(device)
        loss1 = model(images, text, targets=targets, train=True, agent=agent)

        text_, id = generate_new_concept(language_model, tokenizer, text, config, i,agent)
        if agent.LwF or agent.cls:
            loss2 = model(images, text_, targets=targets, train=True, agent=agent,coreset_input=True)
        else:
            loss2 = model(images, text_, wild_id=[id], targets=targets, train=True, agent=agent, train_zsl=True)


        loss = loss1 + loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=loss.item())
        metric_logger.update(our_loss=loss2.item())

        if i % print_freq == 0 and i != 0:
            print(f"Step: {i}, Loss: {loss.item():.4f}, ce-Loss: {loss1.item():.4f}, our_loss: {loss2.item():.4f}")

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def train_cur(model, data_loader,optimizer, epoch, device, config, agent):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 10

    for i, batch_data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        #task data loss cal
        image0, pos, neg, idx = batch_data
        text = pos + neg
        image0 = image0.repeat(2, 1, 1, 1)
        targets = torch.zeros((len(text, )), dtype=torch.int64)
        targets[:len(pos)] = 1
        images = image0
        images, targets = images.to(device), targets.to(device)
        if agent.lwf_ours:
            loss1 = model(images, text, targets=targets, train=True, agent=agent)
            # text_, id = generate_new_concept(language_model, tokenizer, text, config, i,agent)
            id = agent.task_id
            loss2 = model(images, text, wild_id=[id], targets=targets, train=True, agent=agent, train_zsl=True)
            loss = loss1 + loss2
        else:
            loss = model(images, text, targets=targets, train=True, agent=agent, coreset_input=True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=loss.item())
        # metric_logger.update(our_loss=loss2.item())

        # if i % print_freq == 0 and i != 0:
        #     print(f"Step: {i}, Loss: {loss.item():.4f}, ce-Loss: {loss1.item():.4f}, our_loss: {loss2.item():.4f}")

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


def train(model, data_loader, optimizer, epoch, device, config, agent,coreset_loader):
    # train
    model.train()  
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50   
    step_size = 10
    if coreset_loader is not None:
        coreset_loader = cycle(coreset_loader)
        for i, (batch_data, coreset_batch) in enumerate(zip(
                metric_logger.log_every(data_loader, print_freq, header),
                coreset_loader)):  # 用 None 填充短的 DataLoader

            image0, pos, neg, idx = batch_data
            text = pos + neg
            image0 = image0.repeat(2, 1, 1, 1)
            targets = torch.zeros((len(text),), dtype=torch.int64)
            targets[:len(pos)] = 1
            images = image0
            images, targets = images.to(device), targets.to(device)

            coreset_image, coreset_pos, coreset_neg, coreset_idx, sample_id = coreset_batch
            sample_id = list(sample_id)
            coreset_text = list(coreset_pos) + list(coreset_neg)
            coreset_image = coreset_image.repeat(2, 1, 1, 1)
            coreset_targets = torch.zeros((len(coreset_text),), dtype=torch.int64)
            coreset_targets[:len(coreset_pos)] = 1
            coreset_images = coreset_image
            coreset_images, coreset_targets = coreset_images.to(device), coreset_targets.to(device)

            if agent.ours_id:
                loss1 = model(images, text, targets=targets, train=True, agent=agent)
                loss2 = model(coreset_images, coreset_text, targets=coreset_targets, train=True, agent=agent)

                loss_zsl = torch.zeros(1, dtype=torch.float32, device=device)
                wild_id = torch.tensor(sample_id, dtype=torch.int64, device=device)
                unique_wild_ids = torch.unique(wild_id)  # 获取所有 unique 的 wild_id
                num_batches = 0
                for current_wild_id in unique_wild_ids:  # 遍历每个 unique wild_id
                    # 获取当前 wild_id 对应的样本索引
                    current_indices = torch.where(wild_id == current_wild_id)[0]
                    current_images_ = coreset_images[current_indices]
                    current_targets_ = coreset_targets[current_indices]
                    current_id = wild_id[current_indices]
                    current_indices_list = current_indices.tolist()
                    current_text_ = [coreset_text[i] for i in current_indices_list]  # Use list indexing
                    loss_cur = model(current_images_, current_text_, wild_id=current_id, targets=current_targets_, train=True,
                                 agent=agent, train_zsl=True)
                    loss_zsl = loss_zsl + loss_cur
                    num_batches += 1
                loss3 = loss_zsl / num_batches
                loss = loss1 + loss2 + loss3
            else:
                if agent.cls:
                    loss = model(images, text, targets=targets, train=True, agent=agent,coreset_input=False)
                    loss_cls = model(coreset_images, coreset_text, targets=coreset_targets, train=True, agent=agent,coreset_input=True)
                    loss = loss + loss_cls
                elif agent.LwF_ER:
                    loss = model(images, text, targets=targets, train=True, agent=agent, coreset_input=False)
                    loss_LwF_ER = model(coreset_images, coreset_text, targets=coreset_targets, train=True, agent=agent,
                                     coreset_input=True)
                    loss = loss + loss_LwF_ER
                else:
                    # **合并数据**

                    all_images = torch.cat([images, coreset_images],
                                           dim=0)
                    all_text = text + coreset_text
                    all_targets = torch.cat([targets, coreset_targets],
                                            dim=0)

                    if agent.gem and agent.coreset != []:
                        grad_numels = []
                        for n, p in model.named_parameters():
                            if p.requires_grad:
                                grad_numels.append(p.data.numel())
                        G = torch.zeros((sum(grad_numels), agent.task_id + 1)).to(device)
                        compute_gem(model,optimizer,grad_numels,G,agent,device)

                    # **计算 loss**
                    loss = model(all_images, all_text, targets=all_targets, train=True, agent=agent)

                    if agent.ewc and agent.fisher != None:
                        loss_ewc = compute_ewc(model,agent)
                        loss = loss + 1e5 * loss_ewc

            # 反向传播 & 更新参数
            optimizer.zero_grad()
            loss.backward()
            if agent.gem and agent.coreset != []:
                compute_gem_current(model,grad_numels,G,agent,device)
            optimizer.step()

            # 更新日志
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(loss=loss.item())
    else:
        for i, batch_data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            image0, pos, neg, idx = batch_data
            text = pos + neg
            image0 = image0.repeat(2, 1, 1, 1)
            targets = torch.zeros((len(text,)), dtype=torch.int64)
            targets[:len(pos)] = 1
            images = image0
            images, targets = images.to(device), targets.to(device)

            loss = model(images, text, targets=targets, train=True, agent=agent)

            if agent.ewc and agent.fisher != None:
                loss_ewc = compute_ewc(model,agent)
                loss = loss + 1e5 * loss_ewc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(loss=loss.item())
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

def train_zsl(model, data_loader,zsl_data_loader,zsl_datasets,samplers,num_workers,optimizer, epoch, device, config, agent):
    # train
    model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))
    metric_logger.add_meter('zero_shot_loss', utils.SmoothedValue(window_size=50, fmt='{value:.4f}'))

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 10
    zsl_data_loader = cycle(zsl_data_loader)
    for i, (batch_data, zsl_batch_data) in enumerate(
            zip(data_loader, zsl_data_loader)):
        #task data loss cal
        image0, pos, neg, idx = batch_data
        text = pos + neg
        image0 = image0.repeat(2, 1, 1, 1)
        targets = torch.zeros((len(text, )), dtype=torch.int64)
        targets[:len(pos)] = 1
        images = image0
        images, targets = images.to(device), targets.to(device)
        loss1 = model(images, text, targets=targets, train=True, agent=agent)

        if agent.train_distill_type == 'zsl-single' or  agent.train_distill_type == 'ema-zsl-single' or  agent.train_distill_type == 'adv_text_zsl' or agent.train_distill_type == 'multi-zsl-single':
            loss_zsl = torch.zeros(1, dtype=torch.float32, device=device)

            image0, pos, neg, idx, wild_id = zsl_batch_data

            text = pos + neg
            image0 = image0.repeat(2, 1, 1, 1)
            # wild_id = wild_id.repeat(2)
            targets = torch.ones((len(text, )), dtype=torch.int64)#do not invlove in calculation
            images = image0
            images, targets = images.to(device), targets.to(device)
            if agent.zaf_ours:
                # wild_id = [agent.task_id] * len(wild_id)
                loss2 = model(images, text, wild_id=[agent.task_id], targets=targets, train=True, agent=agent, train_zsl=True)
            else:
                wild_id = torch.tensor(wild_id, dtype=torch.int64, device=device)
                unique_wild_ids = torch.unique(wild_id)  # 获取所有 unique 的 wild_id
                num_batches = 0
                for current_wild_id in unique_wild_ids:  # 遍历每个 unique wild_id
                    # 获取当前 wild_id 对应的样本索引
                    current_indices = torch.where(wild_id == current_wild_id)[0]
                    current_images = images[current_indices]
                    current_targets = targets[current_indices]
                    current_id = wild_id[current_indices]
                    current_indices_list = current_indices.tolist()
                    current_text = [text[i] for i in current_indices_list]  # Use list indexing

                    loss = model(current_images, current_text, wild_id=current_id, targets=current_targets, train=True, agent=agent, train_zsl=True)
                    loss_zsl = loss_zsl + loss
                    num_batches += 1


                loss2 = loss_zsl / num_batches
        losses_log = loss2.item()

        loss = loss1 + loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=loss.item())
        metric_logger.update(zero_shot_loss=loss2.item())

        if i % print_freq == 0 and i != 0:
            print(f"Step: {i}, Loss: {loss.item():.4f}, ce-Loss: {loss1.item():.4f}, zero-shot-Loss: {loss2.item():.4f}, task-Loss: {losses_log}")

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model, data_loader, device, config, agent):
    # test
    model.eval()
            
    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Evaluation:'
    print_freq = 50

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        if not isinstance(batch_data[1], list):
            image0, image1, text, targets = batch_data
        else:
            image0, pos, neg, idx = batch_data
            text = pos + neg
            image0 = image0.repeat(2, 1, 1, 1)
            targets = torch.zeros((len(text, )), dtype=torch.int64)
            targets[:len(pos)] = 1
            image1 = None
        if image1 is not None:
            images = torch.cat([image0, image1], dim=0)
        else:
            images = image0
        images, targets = images.to(device), targets.to(device)   
        
        prediction = model(images, text, targets=targets, train=False, agent=agent)  
 
        _, pred_class = prediction.max(1)
        accuracy = (targets==pred_class).sum() / targets.size(0)
        
        metric_logger.meters['acc'].update(accuracy.item(), n=image0.size(0))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print("Averaged stats:", metric_logger.global_avg())   
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def multi_task_evaluate(model, data_loader, device, config, agent):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")

    header = 'Evaluation:'
    print_freq = 50

    for batch_data in metric_logger.log_every(data_loader, print_freq, header):
        if not isinstance(batch_data[1], list):
            image0, image1, text, targets = batch_data
        else:
            image0, pos, neg, idx = batch_data
            text = pos + neg
            image0 = image0.repeat(2, 1, 1, 1)
            targets = torch.zeros((len(text,)), dtype=torch.int64)
            targets[:len(pos)] = 1
            image1 = None
        if image1 is not None:
            images = torch.cat([image0, image1], dim=0)
        else:
            images = image0
        images, targets = images.to(device), targets.to(device)

        predictions = []
        for iT in range(agent.get_num_tasks()):
            agent.prep_model4task(iT)
            prediction = model(images, text, targets=targets, train=False, agent=agent)
            if isinstance(prediction, tuple):
                fuse_weights = prediction[1].detach().cpu()
                prediction = prediction[0]
            predictions.append(prediction.detach().cpu())
        agent.prep_model4task(-1)


        if agent.fuse_type in ['last']:
            prediction = torch.stack([x.softmax(dim=-1) for x in predictions], dim=-1)
        else:
            prediction = torch.stack(predictions, dim=-1)

        if agent.fuse_type in ['last']:
            prediction = prediction[:, :, -1]
        else:
            raise NotImplementedError(f'Unsupported fuse type: {agent.fuse_type}')

        _, pred_class = prediction.max(1)
        accuracy = (targets.cpu() == pred_class).sum() / targets.size(0)

        metric_logger.meters['acc'].update(accuracy.item(), n=image0.size(0))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print("Averaged stats:", metric_logger.global_avg())
    return {k: "{:.4f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}

        
def main(args, config, eval=False,test_ema=False):

    agent = args['agent']


    # change all "args." to args[""]
    args['result_dir'] = os.path.join(args['out_dir'], 'result')
    if utils.is_main_process(): Path(args['result_dir']).mkdir(parents=True, exist_ok=True)
    device = args['device']

    #### Dataset #### 
    print("Creating dataset")
    dataset_pass_dict = {'training_data_sample':args['training_data_sample']}
    datasets = create_dataset(config['dataset'], config, dataset_pass_dict)

    if args['distributed']:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()            
        samplers = create_sampler(datasets, [True,False,False], num_tasks, global_rank)
    else:
        samplers = [None, None, None]
    
    batch_size=[config['batch_size_train'][agent.task_id],config['batch_size_test'],config['batch_size_test']]
    train_loader, val_loader, test_loader = create_loader(datasets,samplers,batch_size=batch_size,
                                                          num_workers=[args['num_workers'], args['num_workers'], args['num_workers']],is_trains=[True,False,False],
                                                          collate_fns=[None,None,None])

    # agent
    agent = args['agent']

    #### Model #### 
    print("Creating model")

    model, head_not_loaded = blip_nlvr(pretrained=args['pretrained'], image_size=config['image_size'],
                         vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], agent=agent, single_image_model=('vl-checklist' in config['dataset']))

    model = model.to(device)
    model_without_ddp = model
    if args['distributed']:
        if agent.CFT:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args['gpu']],find_unused_parameters=False)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args['gpu']], find_unused_parameters=True)
        model_without_ddp = model.module

    if not eval and agent.ema:
        if  args['ema_lora'] == 'continual':
            print('continual training lora ')
            pass
        elif args['ema_lora'] == 'ema':
            print('initial lora with ema')
            if args['model'] == 'blip':
                lora.lora_initial_ema(model_without_ddp.text_encoder)
                lora.lora_initial_ema(model_without_ddp.visual_encoder)
            elif args['model'] == 'beit':
                lora.lora_initial_ema(model_without_ddp.beit3)

    if not eval and agent.slow:
        lora.update_quick_lora_ini_slow(model_without_ddp.text_encoder, agent.task_id)
        lora.update_quick_lora_ini_slow(model_without_ddp.visual_encoder, agent.task_id)

    if agent.freeze_encoders:
        
        param_to_optim = []
        
        # task heads
        if (not agent.freeze_heads) or head_not_loaded:
        # if False:
            print('Training head')
            param_to_optim += list(model_without_ddp.cls_head.parameters())
        else:
            print('Locking head')
            for p in model_without_ddp.cls_head.parameters():
                p.requires_grad = False

        if agent.lora or agent.adapter:
            if args['model'] == 'blip':
                param_to_optim += list(model_without_ddp.text_encoder.parameters())
                param_to_optim += list(model_without_ddp.visual_encoder.parameters())
                lora.mark_only_lora_as_trainable(model_without_ddp.text_encoder)
                lora.mark_only_lora_as_trainable(model_without_ddp.visual_encoder)
            elif args['model'] == 'beit':
                param_to_optim += list(model_without_ddp.beit3.parameters())
                lora.mark_only_lora_as_trainable(model_without_ddp)

        # optimizer
        optimizer = torch.optim.AdamW(params=param_to_optim, lr=config['init_lr'], weight_decay=config['weight_decay'])
        nparam = count_parameters(param_to_optim)
    else:
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
        nparam = count_parameters(model.parameters())

    # print num trainable params    
    print(f'trainable_parameters = {nparam}')


    print("Start training")
    start_time = time.time()
    best = 0
    best_epoch = 0

    # flag for no training
    if not eval and args['eval_every'] < 0:
        if utils.is_main_process():  
            torch.save({'model':model_without_ddp.state_dict()}, args['model_save_path'])
        return

    start_epoch = 0

    #load checkpint of current task
    for epoch in range(start_epoch, config['max_epoch']):
        load_file = os.path.join(args['out_dir'], 'checkpoint_%02d.pth'%epoch)
        if os.path.exists(load_file):
            checkpoint = torch.load(load_file)
            model_without_ddp.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            best = checkpoint['best']
            best_epoch = checkpoint['best_epoch']

    if agent.LM and not eval:
        print('processing concepts')
        extract_concept = extract_nlvr_concept(datasets[0])
        agent.concept_dict[config['all_task_name'][agent.task_id]] = extract_concept

    if test_ema:
        fuse_type = agent.fuse_type
        print('evaluate ema')
        agent = args['agent']
        #agent.fuse_type = 'ema'
        print("Creating model")

        model, head_not_loaded = blip_nlvr(pretrained=args['pretrained'], image_size=config['image_size'],
                                           vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'],
                                           vit_ckpt_layer=config['vit_ckpt_layer'], agent=agent,
                                           single_image_model=('vl-checklist' in config['dataset']))
        model = model.to(device)
        if args['distributed']:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args['gpu']],
                                                              find_unused_parameters=True)

        eval_func = evaluate

        test_stats = eval_func(model, test_loader, device, config, agent)

        agent.fuse_type = fuse_type

        if utils.is_main_process():
            return test_stats['acc']
        else:
            return -0.1

    elif eval and agent.cls:
        fuse_type = agent.fuse_type
        print('evaluate ema')
        agent = args['agent']
        # agent.fuse_type = 'slow'
        print("Creating model")
        model, head_not_loaded = blip_nlvr(pretrained=args['pretrained'], image_size=config['image_size'],
                                           vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'],
                                           vit_ckpt_layer=config['vit_ckpt_layer'], agent=agent,
                                           single_image_model=('vl-checklist' in config['dataset']))
        model = model.to(device)
        if args['distributed']:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args['gpu']],
                                                              find_unused_parameters=True)

        eval_func = evaluate

        test_stats = eval_func(model, test_loader, device, config, agent)

        agent.fuse_type = fuse_type

        if utils.is_main_process():
            return test_stats['acc']
        else:
            return -0.1


    elif eval and agent.slow:

        print('evaluate divide ema')
        agent = args['agent']
        agent.eval = True
        print("Creating model")
        model, head_not_loaded = blip_nlvr(pretrained=args['pretrained'], image_size=config['image_size'],
                                           vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'],
                                           vit_ckpt_layer=config['vit_ckpt_layer'], agent=agent,
                                           single_image_model=('vl-checklist' in config['dataset']))
        model = model.to(device)
        if args['distributed']:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args['gpu']],
                                                              find_unused_parameters=True)

        eval_func = evaluate

        test_stats = eval_func(model, test_loader, device, config, agent)
        agent.eval = False
        if utils.is_main_process():
            return test_stats['acc']
        else:
            return -0.1

    else:
        if agent.task_id != 0 and agent.zaf == True:
            print("Creating zsl dataset")
            dataset_pass_dict = {'training_data_sample': args['training_data_sample']}
            zsl_datasets = create_zsl_dataset(config['dataset'], config, dataset_pass_dict)

            if args['distributed']:
                num_tasks = utils.get_world_size()
                global_rank = utils.get_rank()
                samplers = create_sampler(zsl_datasets, [True, False, False], num_tasks, global_rank)
            else:
                samplers = [None, None, None]

            batch_size = [config['batch_size_train'][agent.task_id], config['batch_size_test'], config['batch_size_test']]
            zsl_train_loader, _, _ = create_loader(zsl_datasets, samplers, batch_size=batch_size,
                                                   num_workers=[args['num_workers'], args['num_workers'],
                                                                args['num_workers']], is_trains=[True, False, False],
                                                   collate_fns=[None, None, None])
        coreset_loader = None
        if agent.coreset != []:
            print("Creating replay dataset")
            coreset_dataset = CoresetDataset(agent.coreset)
            coreset_loader = DataLoader(coreset_dataset, batch_size=config['batch_size_train'][agent.task_id], num_workers=args['num_workers'], shuffle=True,collate_fn=None)

        if agent.LM:
            Language_model,tokenizer = Create_LM(config['LM_path'])

        best = 0
        for epoch in range(start_epoch, config['max_epoch']):
            if not eval:
                if args['distributed']:
                    train_loader.sampler.set_epoch(epoch)

                cosine_lr_schedule(optimizer, epoch, config['max_epoch'], config['init_lr'], config['min_lr'])

                if agent.task_id != 0 and agent.zaf == True:
                    train_stats = train_zsl(model, train_loader,zsl_train_loader,zsl_datasets,samplers,args['num_workers'], optimizer, epoch, device, config, agent)
                elif agent.task_id != 0 and agent.LM:
                    train_stats = train_ours(model, train_loader, Language_model, tokenizer,optimizer, epoch, device, config, agent)
                elif agent.task_id != 0  and (agent.lwf_ours or agent.cur_distill):
                    train_stats = train_cur(model, train_loader,optimizer, epoch, device, config, agent)
                else:
                    train_stats = train(model, train_loader, optimizer, epoch,  device, config, agent,coreset_loader)

                if agent.ema and (epoch + 1) % args['ema_frequency'] == 0:
                    frequency = args['ema_frequency']
                    ema_alpha = args['ema_alpha']
                    if args['model'] == 'blip':
                        print(f'epoch EMA begins,current_alpha = {ema_alpha},task_id = {agent.task_id},ema_frequency = {frequency}')
                        lora.update_ema_epoch_lora(model_without_ddp.text_encoder, args['ema_alpha'], agent.task_id,agent.update_both)
                        lora.update_ema_epoch_lora(model_without_ddp.visual_encoder, args['ema_alpha'], agent.task_id,agent.update_both)
                    elif args['model'] == 'beit':
                        lora.update_ema_epoch_lora(model_without_ddp.beit3, args['ema_alpha'],
                                                   agent.task_id, agent.update_both)

                    val_stats = evaluate(model, val_loader, device, config, agent)
                    if args['save_frequency'] == 'best':
                        if float(val_stats['acc']) > best:
                            best = float(val_stats['acc'])
                            torch.save({'model': model_without_ddp.state_dict()}, args['model_save_path'])
                    elif args['save_frequency'] == 'every':
                        torch.save({'model': model_without_ddp.state_dict()}, args['model_save_path'])
                elif agent.cls and (epoch + 1) % args['ema_frequency'] == 0:
                    frequency = args['ema_frequency']
                    ema_alpha = args['ema_alpha']
                    if args['model'] == 'blip':
                        print(f'epoch EMA begins,current_alpha = {ema_alpha},task_id = {agent.task_id},ema_frequency = {frequency}')
                        lora.update_ema_epoch_lora_cls(model_without_ddp.text_encoder, args['ema_alpha'], agent.task_id,agent.update_both)
                        lora.update_ema_epoch_lora_cls(model_without_ddp.visual_encoder, args['ema_alpha'], agent.task_id,agent.update_both)
                    elif args['model'] == 'beit':
                        lora.update_ema_epoch_lora_cls(model_without_ddp.beit3, args['ema_alpha'],
                                                   agent.task_id, agent.update_both)

                    val_stats = evaluate(model, val_loader, device, config, agent)
                    if args['save_frequency'] == 'best':
                        if float(val_stats['acc']) > best:
                            best = float(val_stats['acc'])
                            torch.save({'model': model_without_ddp.state_dict()}, args['model_save_path'])
                    elif args['save_frequency'] == 'every':
                        torch.save({'model': model_without_ddp.state_dict()}, args['model_save_path'])

                if agent.ewc and epoch == config['max_epoch'] - 1:
                    optimizer = torch.optim.AdamW(params=param_to_optim, lr=config['init_lr'],
                                                  weight_decay=config['weight_decay'])
                    if agent.fisher is None:
                        agent.fisher = getFisherDiagonal(model, train_loader, optimizer, device, agent)
                    else:
                        new_fisher = getFisherDiagonal(model, train_loader, optimizer, device, agent)
                        for n, p in new_fisher.items():
                            new_fisher[n][: len(agent.fisher[n])] = (
                                    0.5 * agent.fisher[n]
                                    + 0.5 * new_fisher[n][: len(agent.fisher[n])]
                            )
                        agent.fisher = new_fisher
                    agent.mean = {
                        n: p.clone().detach()
                        for n, p in model.named_parameters()
                        if p.requires_grad
                    }

                if agent.slow and agent.task_id!=0:
                    frequency = args['ema_frequency']
                    ema_alpha = args['ema_alpha']
                    print(
                        f'epoch divide EMA begins,current_alpha = {ema_alpha},task_id = {agent.task_id},ema_frequency = {frequency}')
                    lora.update_ema_epoch_lora_slow(model_without_ddp.text_encoder, args['ema_alpha'], agent.task_id,epoch)
                    lora.update_ema_epoch_lora_slow(model_without_ddp.visual_encoder, args['ema_alpha'],agent.task_id,epoch)

                if agent.LwF and epoch == config['max_epoch'] - 1:
                    print('initial distillation lora with training lora')
                    lora.lora_initial_lora(model_without_ddp.text_encoder)
                    lora.lora_initial_lora(model_without_ddp.visual_encoder)
                    torch.save({'model': model_without_ddp.state_dict()}, args['model_save_path'])

            if eval or (epoch + 1) % args['eval_every'] == 0:
                eval_func = evaluate
                # if agent.multi:
                #     eval_func = multi_task_evaluate

                val_stats = eval_func(model, val_loader, device, config, agent)
                test_stats = eval_func(model, test_loader, device, config, agent)

                if utils.is_main_process():
                    if eval:
                        log_stats = {**{f'val_{k}': v for k, v in val_stats.items()},
                                     **{f'test_{k}': v for k, v in test_stats.items()},
                                     }
                        with open(os.path.join(args['out_dir'], "log.txt"), "a") as f:
                            f.write(json.dumps(log_stats) + "\n")

                    else:
                        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                                     **{f'val_{k}': v for k, v in val_stats.items()},
                                     **{f'test_{k}': v for k, v in test_stats.items()},
                                     'epoch': epoch,
                                     }


                        if float(val_stats['acc']) > best :
                            best = float(val_stats['acc'])
                            best_epoch = epoch
                            if not agent.ema:
                                if args['save_frequency'] == 'best':
                                    torch.save({'model': model_without_ddp.state_dict()}, args['model_save_path'])

                        if args['save_frequency'] == 'every':
                            torch.save({'model': model_without_ddp.state_dict()}, args['model_save_path'])


                        with open(os.path.join(args['out_dir'], "log.txt"), "a") as f:
                            f.write(json.dumps(log_stats) + "\n")

                        save_obj = {
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'config': config,
                            'epoch': epoch,
                            'best': best,
                            'best_epoch': best_epoch,
                        }
                        torch.save(save_obj, os.path.join(args['out_dir'], 'checkpoint_%02d.pth' % epoch))
                        epoch_old = epoch - 1
                        old_file = os.path.join(args['out_dir'], 'checkpoint_%02d.pth' % epoch_old)
                        if os.path.isfile(old_file):
                            os.remove(old_file)

                        print(f'Finished epoch {epoch} best epoch is {best_epoch} with acc {best}')

            dist.barrier()
            torch.cuda.empty_cache()
            if eval:
                if utils.is_main_process():
                    return test_stats['acc']
                else:
                    return -0.1


def sample_memory(memory_size, args,config):
    agent = args['agent']
    #### Dataset ####
    print("Update Memory")
    dataset_pass_dict = {'training_data_sample': args['training_data_sample']}
    datasets = create_dataset(config['dataset'], config, dataset_pass_dict)

    if args['distributed']:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(datasets, [True, False, False], num_tasks, global_rank)
    else:
        samplers = [None, None, None]

    batch_size = [1, 1, 1]
    train_loader, val_loader, test_loader = create_loader(datasets, samplers, batch_size=batch_size,
                                                          num_workers=[args['num_workers'], args['num_workers'],
                                                                       args['num_workers']],
                                                          is_trains=[True, False, False],
                                                          collate_fns=[None, None, None])

    for i, batch_data in enumerate(train_loader):
        image0, pos, neg, idx = batch_data
        # if len(agent.coreset) < memory_size:
        if i < memory_size:
           agent.coreset.append((image0[0], pos[0], neg[0], idx[0], agent.task_id + 1))
           # agent.coreset.append((image0[0], pos[0], neg[0], idx[0]))
        else:
            break
class CoresetDataset(Dataset):
    def __init__(self, coreset):
        self.coreset = coreset  # 直接存储 coreset 数据

    def __len__(self):
        return len(self.coreset)

    def __getitem__(self, idx):
        return self.coreset[idx]  # 返回单个样本



def getFisherDiagonal(model, train_loader, optimizer,device,agent):
    fishermax = 0.0001
    fisher = {
        n: torch.zeros(p.shape).to(model.device)
        for n, p in model.named_parameters()
        if p.requires_grad
    }
    model.train()
    for i, batch_data in enumerate(train_loader):
        image0, pos, neg, idx = batch_data
        text = pos + neg
        image0 = image0.repeat(2, 1, 1, 1)
        targets = torch.zeros((len(text),), dtype=torch.int64)
        targets[:len(pos)] = 1
        images = image0
        images, targets = images.to(device), targets.to(device)
        loss = model(images, text, targets=targets, train=True, agent=agent)
        optimizer.zero_grad()
        loss.backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.pow(2).clone()
    for n, p in fisher.items():
        fisher[n] = p / len(train_loader)
        fisher[n] = torch.min(fisher[n], torch.tensor(fishermax))
    return fisher


def update_fisher(agent):
    params_0 = {}
    params_1 = {}
    last_id = agent.task_id
    cur_id = agent.task_id + 1
    # 第一遍遍历：收集lora0的参数
    for name, param in agent.fisher.items():
        if f'lora_A.{last_id}' in name:
            # 以新的键存储参数，将'.0'替换为'.1'以便后续匹配lora1
            params_0[name.replace(f'lora_A.{last_id}', f'lora_A.{cur_id}')] = param.clone()
        if f'lora_B.{last_id}' in name:
            # 以新的键存储参数，将'.0'替换为'.1'以便后续匹配lora1
            params_0[name.replace(f'lora_B.{last_id}', f'lora_B.{cur_id}')] = param.clone()
        if f'adapter.{last_id}' in name:
            params_0[name.replace(f'adapter.{last_id}', f'adapter.{cur_id}')] = param.clone()

    for name, param in agent.mean.items():
        if f'lora_A.{last_id}' in name:
            # 以新的键存储参数，将'.0'替换为'.1'以便后续匹配lora1
            params_1[name.replace(f'lora_A.{last_id}', f'lora_A.{cur_id}')] = param.clone()
        if f'lora_B.{last_id}' in name:
            # 以新的键存储参数，将'.0'替换为'.1'以便后续匹配lora1
            params_1[name.replace(f'lora_B.{last_id}', f'lora_B.{cur_id}')] = param.clone()
        if f'adapter.{last_id}' in name:
            params_1[name.replace(f'adapter.{last_id}', f'adapter.{cur_id}')] = param.clone()

    return params_0,params_1


def update_fisher_1(agent):
    params_0 = {}
    params_1 = {}
    cur_id = agent.task_id + 1
    # 第一遍遍历：收集lora0的参数
    for name, param in agent.fisher.items():
        if f'lora_A' in name:
            # 以新的键存储参数，将'.0'替换为'.1'以便后续匹配lora1
            params_0[name.replace(f'lora_A', f'lora_A.{cur_id}')] = param.clone()
        if f'lora_B' in name:
            # 以新的键存储参数，将'.0'替换为'.1'以便后续匹配lora1
            params_0[name.replace(f'lora_B', f'lora_B.{cur_id}')] = param.clone()
        if f'adapter.{0}' in name:
            params_0[name.replace(f'adapter.{0}', f'adapter.{cur_id}')] = param.clone()
    # 第一遍遍历：收集lora0的参数
    for name, param in agent.mean.items():
        if f'lora_A' in name:
            # 以新的键存储参数，将'.0'替换为'.1'以便后续匹配lora1
            params_1[name.replace(f'lora_A', f'lora_A.{cur_id}')] = param.clone()
        if f'lora_B' in name:
            # 以新的键存储参数，将'.0'替换为'.1'以便后续匹配lora1
            params_1[name.replace(f'lora_B', f'lora_B.{cur_id}')] = param.clone()
        if f'adapter.{0}' in name:
            params_1[name.replace(f'adapter.{0}', f'adapter.{cur_id}')] = param.clone()

    return params_0,params_1

def compute_ewc(model,agent):
    loss = 0
    for n, p in model.named_parameters():
        if n in agent.fisher.keys():
            loss += (
                torch.sum(
                    (agent.fisher[n])
                    * (p[: len(agent.mean[n])] - agent.mean[n]).pow(2)
                )
                / 2
            )
    return loss

def compute_gem(model,optimizer,grad_numels,G,agent,device):
    for k in range(0, agent.task_id):
        optimizer.zero_grad()
        begin = k * agent.memory_size
        end = (k+1) * agent.memory_size
        data_ = agent.coreset[begin:end]
        image0_list = [item[0] for item in data_]
        pos_list = [item[1] for item in data_]
        neg_list = [item[2] for item in data_]
        idx_list = [item[3] for item in data_]

        image0 = torch.stack(image0_list, dim=0)
        text = pos_list + neg_list
        image0 = image0.repeat(2, 1, 1, 1)
        targets = torch.zeros((len(text),), dtype=torch.int64)
        targets[:len(pos_list)] = 1
        images = image0
        # images, targets = images.to(device), targets.to(device)
        # **计算 loss**
        loss = model(images, text, targets=targets, train=True, agent=agent)
        loss.backward()

        j = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                if p is not None:
                    if j == 0:
                        stpt = 0
                    else:
                        stpt = sum(grad_numels[:j])
                    endpt = sum(grad_numels[: j + 1])
                    G[stpt:endpt, k].data.copy_(p.grad.data.view(-1))
                    j += 1
        optimizer.zero_grad()


def compute_gem_current(model,grad_numels,G,agent,device):
    j = 0
    for n, params in model.named_parameters():
        if params.requires_grad:
            if params is not None:
                if j == 0:
                    stpt = 0
                else:
                    stpt = sum(grad_numels[:j])
                endpt = sum(grad_numels[: j + 1])
                G[stpt:endpt, agent.task_id].data.copy_(
                    params.grad.data.view(-1)
                )
                j += 1

    dotprod = torch.mm(
        G[:, agent.task_id].unsqueeze(0), G[:, : agent.task_id]
    )

    if (dotprod < 0).sum() > 0:

        old_grad = G[:, : agent.task_id].cpu().t().double().numpy()
        cur_grad = G[:, agent.task_id].cpu().contiguous().double().numpy()

        C = old_grad @ old_grad.T
        p = old_grad @ cur_grad
        A = np.eye(old_grad.shape[0])
        b = np.zeros(old_grad.shape[0])

        v = solve_qp(C, -p, A, b)[0]

        new_grad = old_grad.T @ v + cur_grad
        new_grad = torch.tensor(new_grad).float().to(device)

        new_dotprod = torch.mm(
            new_grad.unsqueeze(0), G[:, : agent.task_id]
        )
        if (new_dotprod < -0.01).sum() > 0:
            assert 0

        j = 0
        for n, params in model.named_parameters():
            if params.requires_grad:
                if params is not None:
                    if j == 0:
                        stpt = 0
                    else:
                        stpt = sum(grad_numels[:j])
                    endpt = sum(grad_numels[: j + 1])
                    params.grad.data.copy_(
                        new_grad[stpt:endpt]
                        .contiguous()
                        .view(params.grad.data.size())
                    )
                    j += 1