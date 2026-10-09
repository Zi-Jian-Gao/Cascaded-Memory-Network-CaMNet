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
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from data.vqa_dataset import vqa_collate_fn,vqa_collate_fn_zsl
from models.blip_vqa import blip_vqa
from quadprog import solve_qp
import utils
from utils import cosine_lr_schedule, warmup_lr_schedule, count_parameters
from data import create_dataset, create_sampler, create_loader, create_zsl_dataset
from data.vqa_eval import VQAEval
import loralib as lora
import re
from torch.utils.data import Dataset, DataLoader
from task_trainers.LM import Create_LM,generate_new_questions
from itertools import cycle

vqa_tool = VQAEval()

def pre_answer(answer):
    answer = str(answer)
    answer = re.sub(
        r"([,.'!?\"()*#:;~])",
        '',
        answer.lower(),
    ).replace(' \t', ' ')
    answer = answer.replace('x ray', 'xray').replace('x-ray', 'xray')
    answer = answer.replace(' - ', '-')
    return answer


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
        image, question, answer, weights, n = batch_data
        image, weights = image.to(device), weights.to(device)

        loss1 = model(image, question, answer, train=True, n=n, weights=weights, agent=agent)

        #generate question and id
        question_, id = generate_new_questions(language_model,tokenizer,question,config,i)

        if agent.LwF or agent.cls:
            loss2 = model(image, question_, answer, train=True, n=n, weights=weights, agent=agent,coreset_input=True,LM_replay=True)
            # loss2 = model(image, question, answer, train=True, n=n, weights=weights, train_zsl=True, agent=agent, wild_id=[id])
        else:
            loss2 = model(image, question_, answer, train=True, n=n, weights=weights, train_zsl=True, agent=agent, wild_id=[id])


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
        image, question, answer, weights, n = batch_data
        image, weights = image.to(device), weights.to(device)

        #generate question and id
        # question_, id = generate_new_questions(language_model,tokenizer,question,config,i)
        if agent.lwf_ours:
            loss = model(image, question, answer, train=True, n=n, weights=weights, agent=agent)
            id = agent.task_id
            loss2 = model(image, question, answer, train=True, n=n, weights=weights, train_zsl=True, agent=agent, wild_id=[id])
            loss = loss +  0.25 * loss2
        else:
            loss = model(image, question, answer, train=True, n=n, weights=weights, agent=agent,coreset_input=True)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=loss.item())
        # metric_logger.update(our_loss=loss2.item())

        if i % print_freq == 0 and i != 0:
            print(f"Step: {i}, Loss: {loss.item():.4f}, ce-Loss: {loss.item():.4f}")
            # print(f"Step: {i}, Loss: {loss.item():.4f}, ce-Loss: {loss.item():.4f}, our_loss: {loss2.item():.4f}")

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
    from itertools import zip_longest  # 确保 zip_longest 可用

    if coreset_loader is not None:
        coreset_loader = cycle(coreset_loader)
        for i, (batch_data, coreset_batch) in enumerate(zip(
                metric_logger.log_every(data_loader, print_freq, header),
                coreset_loader)):  # 只有 coreset_batch 可能为 None

            # **处理 data_loader 数据**
            image, question, answer, weights, n = batch_data
            image, weights = image.to(device), weights.to(device)

            # **处理 coreset_loader 数据**
            coreset_image, coreset_question, coreset_answer, coreset_weights, coreset_n, sample_id = coreset_batch
            coreset_image, coreset_weights = coreset_image.to(device), coreset_weights.to(device)


            if agent.ours_id:
                coreset_n = list(coreset_n)
                sample_id = list(sample_id)
                loss1 = model(image, question, answer, train=True, n=n, weights=weights, agent=agent)
                loss2 = model(coreset_image, coreset_question, coreset_answer, train=True, n=coreset_n, weights=coreset_weights, agent=agent)

                loss_zsl = torch.zeros(1, dtype=torch.float32, device=device)
                wild_id_tensor = torch.tensor(sample_id, dtype=torch.int64, device=device)
                unique_wild_ids = torch.unique(wild_id_tensor)  # 获取所有 unique 的 wild_id
                num_batches = 0
                for current_wild_id in unique_wild_ids:  # 遍历每个 unique wild_id
                    current_indices = torch.where(wild_id_tensor == current_wild_id)[0]
                    current_image_ = coreset_image[current_indices]
                    current_weight_ = coreset_weights[current_indices]
                    current_indices_list = current_indices.tolist()
                    current_question_ = [coreset_question[i] for i in current_indices_list]
                    current_answer_ = [coreset_answer[i] for i in current_indices_list]
                    current_n_ = [coreset_n[i] for i in current_indices_list]
                    current_id_ = [sample_id[i] for i in current_indices_list]

                    loss_cur = model(current_image_, current_question_, current_answer_, train=True, n=current_n_,
                                 weights=current_weight_, train_zsl=True, agent=agent, wild_id=current_id_)
                    loss_zsl = loss_zsl + loss_cur
                    num_batches += 1
                loss3 = loss_zsl / num_batches
                loss = loss1 + loss2 + loss3
            else:
                if agent.cls:
                    loss = model(image, question, answer, train=True, n=n, weights=weights, agent=agent,coreset_input=False)
                    loss_cls = model(coreset_image, coreset_question, coreset_answer, train=True, n=list(coreset_n), weights=coreset_weights, agent=agent,coreset_input=True)
                    loss = loss + loss_cls
                elif agent.LwF_ER:
                    loss = model(image, question, answer, train=True, n=n, weights=weights, agent=agent,coreset_input=False)
                    loss_LwF_ER = model(coreset_image, coreset_question, coreset_answer, train=True, n=list(coreset_n), weights=coreset_weights, agent=agent,coreset_input=True)
                    loss = loss + loss_LwF_ER
                else:
                    all_images = torch.cat([image, coreset_image], dim=0)
                    all_questions = question + list(coreset_question)
                    all_answers = answer + list(coreset_answer)
                    all_weights = torch.cat([weights, coreset_weights], dim=0)
                    all_n = n + list(coreset_n)

                    if agent.gem and agent.coreset != []:
                        grad_numels = []
                        for n, p in model.named_parameters():
                            if p.requires_grad:
                                grad_numels.append(p.data.numel())
                        G = torch.zeros((sum(grad_numels), agent.task_id + 1)).to(device)
                        compute_gem(model,optimizer,grad_numels,G,agent,device)

                    # **计算 loss**
                    loss = model(all_images, all_questions, all_answers, train=True, n=all_n, weights=all_weights, agent=agent)

                    if agent.ewc and agent.fisher != None:
                        loss_ewc = compute_ewc(model,agent)
                        # print('loss_ewc')
                        # print(loss_ewc)
                        loss = loss + 1e5 *  loss_ewc

            # 反向传播 & 更新参数
            optimizer.zero_grad()
            loss.backward()
            if agent.gem and agent.coreset != []:
                compute_gem_current(model,grad_numels,G,agent,device)
            optimizer.step()

            # 更新日志
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(loss=loss.item())

    else:  # 只有 data_loader 训练
        for i, batch_data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            image, question, answer, weights, n = batch_data
            image, weights = image.to(device), weights.to(device)

            loss = model(image, question, answer, train=True, n=n, weights=weights, agent=agent)

            if agent.ewc and agent.fisher != None:
                loss_ewc = compute_ewc(model,agent)
                # print('loss_ewc')
                # print(loss_ewc)
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
        if all(item is None for item in batch_data) :
            break
        if all(item is None for item in zsl_batch_data):
            break

        image, question, answer, weights, n = batch_data
        image, weights = image.to(device), weights.to(device)

        loss1 = model(image, question, answer, train=True, n=n, weights=weights, agent=agent)

        if agent.train_distill_type == 'zsl-single' or  agent.train_distill_type == 'ema-zsl-single' or  agent.train_distill_type == 'adv_text_zsl' or agent.train_distill_type == 'multi-zsl-single':
            # zsl data loss cal

            image, question, answer, weights, n, wild_id = zsl_batch_data


            image, weights = image.to(device), weights.to(device)
            loss_zsl = torch.zeros(1, dtype=torch.float32, device=device)

            # loss2 = model(image, question, answer, train=True, n=n,
            #              weights=weights, train_zsl=True, agent=agent, wild_id=wild_id)
            if agent.zaf_ours:
                # wild_id = [agent.task_id] * len(wild_id)
                loss2 = model(image, question, answer, train=True, n=n, weights=weights, train_zsl=True, agent=agent, wild_id=[agent.task_id])
            else:
                wild_id_tensor = torch.tensor(wild_id, dtype=torch.int64, device=device)
                unique_wild_ids = torch.unique(wild_id_tensor)  # 获取所有 unique 的 wild_id
                num_batches = 0

                for current_wild_id in unique_wild_ids:  # 遍历每个 unique wild_id
                    current_indices = torch.where(wild_id_tensor == current_wild_id)[0]

                    current_images = image[current_indices]
                    current_weights = weights[current_indices]

                    current_indices_list = current_indices.tolist()
                    current_question = [question[i] for i in current_indices_list]
                    current_answer = [answer[i] for i in current_indices_list]
                    current_n = [n[i] for i in current_indices_list]
                    current_id = [wild_id[i] for i in current_indices_list]

                    loss = model(current_images, current_question, current_answer, train=True, n=current_n,
                                  weights=current_weights, train_zsl=True, agent=agent, wild_id=current_id)

                    loss_zsl = loss_zsl + loss
                    num_batches += 1

                # loss2 = model(image, question, answer, train=True, n=n, weights=weights, train_zsl=True, agent=agent,wild_id=wild_id)

                loss2 = loss_zsl / num_batches
        loss = loss1 + loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss=loss.item())
        metric_logger.update(zero_shot_loss=loss2.item())

        if i % print_freq == 0 and i != 0:
            print(f"Step: {i}, Loss: {loss.item():.4f}, ce-Loss: {loss1.item():.4f}, zero-shot-Loss: {loss2.item():.4f}")

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

    for i, (image, question, answer, weights, n) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
    # for n, (image, question, question_id) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        image = image.to(device)
        gt_answer = answer
        if config['inference'] == 'generate':
            predicted_answers = model(image, question, train=False, inference='generate')

            # 初始化保存准确率的列表
            accQA = []

            # 遍历当前 batch 中的每个样本
            for batch_idx in range(image.size(0)):
                resAns = predicted_answers[batch_idx]

                # 如果预测答案是列表，取第一个元素
                if isinstance(resAns, list):
                    resAns = resAns[0]

                # 处理预测答案中的标点符号、数字和冠词
                resAns = resAns.replace("\n", " ").replace("\t", " ").strip()
                resAns = vqa_tool.processPunctuation(resAns)
                resAns = vqa_tool.processDigitArticle(resAns)
                if config['dataset'] == 'pathvqa':
                    resAns = pre_answer(resAns)


                #这个时候image/question/answer都是一对一的，可做判断条件
                if len(gt_answer) == image.size(0):
                    gtAnswers = gt_answer[batch_idx]
                else:
                # 处理 ground truth 答案
                    gtAnswers = []
                    gtAcc = []

                    #取出对应的gt_answer
                    # 将 ground truth 答案中的每个答案进行标点处理
                    start = 0
                    for i in n[:batch_idx]:
                        start += i
                    end = 0
                    for i in n[:batch_idx+1]:
                        end += i

                    for ansDic in gt_answer[start:end]:
                        gtAnswers.append(vqa_tool.processPunctuation(ansDic))

                    gt_weight = weights[start:end]
                    # 根据weight来还原到原来的10个answer
                    new_answers = []

                    for answer, weight in zip(gtAnswers, gt_weight):
                        # 根据权重复制相应次数，int(weight * 10)会给出复制次数
                        num_copies = int(weight * 10)
                        # 如果权重为0.1，则只保留一次，否则复制多次
                        new_answers.extend([answer] * num_copies)
                    gtAnswers = new_answers

                if isinstance(gtAnswers, str):
                    gtAnswer = gtAnswers
                    gtAnswer = vqa_tool.processPunctuation(gtAnswer)

                    # 如果预测答案与 ground truth 相同
                    if resAns == gtAnswer:
                        accQA.append(1.0)
                    else:
                        accQA.append(0.0)
                else:
                    # 对每个 ground truth 答案，计算匹配度
                    for gtAnsDatum in gtAnswers:
                        otherGTAns = copy.deepcopy(gtAnswers)
                        otherGTAns.remove(gtAnsDatum)
                        # 匹配预测答案与其他 ground truth 答案
                        matchingAns = [item for item in otherGTAns if item == resAns]
                        # VQA 准确率计算公式
                        acc = min(1, float(len(matchingAns)) / 3)
                        gtAcc.append(acc)
                    # 计算当前样本的平均准确率
                    if gtAcc:
                        avgGTAcc = float(sum(gtAcc)) / len(gtAcc)
                        accQA.append(avgGTAcc)
                    else:
                        accQA.append(0)

            # 计算当前 batch 的平均准确率
            batch_accuracy = sum(accQA) / len(accQA)

            metric_logger.meters['acc'].update(batch_accuracy, n=image.size(0))

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
                                                          collate_fns=[vqa_collate_fn,vqa_collate_fn,vqa_collate_fn])

    # agent
    agent = args['agent']

    #### Model #### 
    print("Creating model")
    model, head_not_loaded = blip_vqa(pretrained=args['pretrained'], image_size=config['image_size'],
                         vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], agent=agent)

    model = model.to(device)   
    
    model_without_ddp = model
    if args['distributed']:
        if args['distributed']:
            if agent.CFT:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args['gpu']],
                                                                  find_unused_parameters=False)
            else:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args['gpu']],
                                                                  find_unused_parameters=True)
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args['gpu']], find_unused_parameters=True)
        model_without_ddp = model.module

    if not eval and agent.ema:
        if  args['ema_lora'] == 'continual':
            print('continual training lora ')
            pass
        elif args['ema_lora'] == 'ema':
            print('initial lora with ema')
            lora.lora_initial_ema(model_without_ddp.text_encoder)
            lora.lora_initial_ema(model_without_ddp.visual_encoder)
            lora.lora_initial_ema(model_without_ddp.text_decoder)

    if not eval and agent.cls:
        print('initial lora with learner')
        lora.ini_lora_cls(model_without_ddp.text_encoder)
        lora.ini_lora_cls(model_without_ddp.visual_encoder)
        lora.ini_lora_cls(model_without_ddp.text_decoder)

    if not eval and agent.slow:
        lora.update_quick_lora_ini_slow(model_without_ddp.text_encoder, agent.task_id)
        lora.update_quick_lora_ini_slow(model_without_ddp.visual_encoder, agent.task_id)
        lora.update_quick_lora_ini_slow(model_without_ddp.text_decoder, agent.task_id)

    if agent.freeze_encoders:
        
        param_to_optim = []

        if agent.lora or agent.adapter:
            param_to_optim += list(model_without_ddp.text_encoder.parameters())
            param_to_optim += list(model_without_ddp.visual_encoder.parameters())
            param_to_optim += list(model_without_ddp.text_decoder.parameters())
            lora.mark_only_lora_as_trainable(model_without_ddp.text_encoder)
            lora.mark_only_lora_as_trainable(model_without_ddp.visual_encoder)
            #here we lock the head
            lora.mark_only_lora_as_trainable(model_without_ddp.text_decoder)

        # task heads
        if agent.task_id == 0:
            print('Training head')
            #set require_gradient
            for p in model_without_ddp.text_decoder.cls.parameters():
                p.requires_grad = True
        else:
            print('Locking head')
            for p in model_without_ddp.text_decoder.cls.parameters():
                p.requires_grad = False

        # optimizer
        optimizer = torch.optim.AdamW(params=param_to_optim, lr=config['init_lr'], weight_decay=config['weight_decay'])
        nparam = count_parameters(param_to_optim)
    else:
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
        nparam = count_parameters(model.parameters())


    # print num trainable params    
    print(f'trainable_parameters = {nparam}')
    # for n, p in model.named_parameters():
    #     if  p.requires_grad:
    #         print(f"Parameter Name: {n}, Shape: {p.shape}")



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
            start_epoch = checkpoint['epoch']
            best = checkpoint['best']
            best_epoch = checkpoint['best_epoch']

    if test_ema:
        fuse_type = agent.fuse_type
        print('evaluate ema')
        agent = args['agent']
        #agent.fuse_type = 'ema'
        print("Creating model")
        model, head_not_loaded = blip_vqa(pretrained=args['pretrained'], image_size=config['image_size'],
                                           vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'],
                                           vit_ckpt_layer=config['vit_ckpt_layer'], agent=agent)

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
        model, head_not_loaded = blip_vqa(pretrained=args['pretrained'], image_size=config['image_size'],
                                           vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'],
                                           vit_ckpt_layer=config['vit_ckpt_layer'], agent=agent)

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
        model, head_not_loaded = blip_vqa(pretrained=args['pretrained'], image_size=config['image_size'],
                                          vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'],
                                          vit_ckpt_layer=config['vit_ckpt_layer'], agent=agent)

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
            zsl_train_loader, _, _ = create_loader(zsl_datasets,samplers,batch_size=batch_size,
                                                          num_workers=[args['num_workers'], args['num_workers'], args['num_workers']],is_trains=[True,False,False],
                                                          collate_fns=[vqa_collate_fn_zsl,vqa_collate_fn_zsl,vqa_collate_fn_zsl])
        coreset_loader = None
        if agent.coreset != []:
            print("Creating replay dataset")
            coreset_dataset = CoresetDataset(agent.coreset)
            coreset_loader = DataLoader(coreset_dataset, batch_size=config['batch_size_train'][agent.task_id], num_workers=args['num_workers'], shuffle=True,collate_fn=None)

        if agent.LM:
            Language_model,tokenizer = Create_LM(config['LM_path'])
            #需要告诉他过去有哪些任务名字，然后便利


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
                elif agent.task_id != 0 and (agent.lwf_ours or agent.cur_distill):
                    train_stats = train_cur(model, train_loader,optimizer, epoch, device, config, agent)
                else:
                    train_stats = train(model, train_loader, optimizer, epoch,  device, config, agent,coreset_loader)

                if agent.ema and (epoch + 1) % args['ema_frequency'] == 0:
                    frequency = args['ema_frequency']
                    if args['ema'] == 'epoch':
                        ema_alpha = args['ema_alpha']
                        print(f'epoch EMA begins,current_alpha = {ema_alpha},task_id = {agent.task_id},ema_frequency = {frequency}')
                        lora.update_ema_epoch_lora(model_without_ddp.text_encoder, args['ema_alpha'], agent.task_id,agent.update_both)
                        lora.update_ema_epoch_lora(model_without_ddp.visual_encoder, args['ema_alpha'], agent.task_id,agent.update_both)
                        lora.update_ema_epoch_lora(model_without_ddp.text_decoder, args['ema_alpha'], agent.task_id,agent.update_both)

                        val_stats = evaluate(model, val_loader, device, config, agent)
                        if args['save_frequency'] == 'best':
                            if float(val_stats['acc']) > best:
                                best = float(val_stats['acc'])
                                torch.save({'model': model_without_ddp.state_dict()}, args['model_save_path'])
                        elif args['save_frequency'] == 'every':
                            torch.save({'model': model_without_ddp.state_dict()}, args['model_save_path'])

                elif agent.cls and (epoch + 1) % args['ema_frequency'] == 0:
                    frequency = args['ema_frequency']
                    if args['ema'] == 'epoch':
                        ema_alpha = args['ema_alpha']
                        print(f'epoch EMA begins,current_alpha = {ema_alpha},task_id = {agent.task_id},ema_frequency = {frequency}')
                        lora.update_ema_epoch_lora_cls(model_without_ddp.text_encoder, args['ema_alpha'], agent.task_id,agent.update_both)
                        lora.update_ema_epoch_lora_cls(model_without_ddp.visual_encoder, args['ema_alpha'], agent.task_id,agent.update_both)
                        lora.update_ema_epoch_lora_cls(model_without_ddp.text_decoder, args['ema_alpha'], agent.task_id,agent.update_both)

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

                if agent.LwF and epoch == config['max_epoch'] - 1:
                    print('initial distillation lora with training lora')
                    lora.lora_initial_lora(model_without_ddp.text_encoder)
                    lora.lora_initial_lora(model_without_ddp.text_decoder)
                    lora.lora_initial_lora(model_without_ddp.visual_encoder)
                    torch.save({'model': model_without_ddp.state_dict()}, args['model_save_path'])


                if agent.slow and agent.task_id!=0:
                    frequency = args['ema_frequency']
                    ema_alpha = args['ema_alpha']
                    print(
                        f'epoch divide EMA begins,current_alpha = {ema_alpha},task_id = {agent.task_id},ema_frequency = {frequency}')
                    lora.update_ema_epoch_lora_slow(model_without_ddp.text_encoder, args['ema_alpha'], agent.task_id,epoch)
                    lora.update_ema_epoch_lora_slow(model_without_ddp.visual_encoder, args['ema_alpha'],agent.task_id,epoch)
                    lora.update_ema_epoch_lora_slow(model_without_ddp.text_decoder, args['ema_alpha'],agent.task_id,epoch)



            if eval or (epoch + 1) % args['eval_every'] == 0:
                eval_func = evaluate

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
                                                          collate_fns=[vqa_collate_fn, vqa_collate_fn, vqa_collate_fn])

    for i, batch_data in enumerate(train_loader):
        image, question, answer, weights, n = batch_data
        #最大2000
        # if len(agent.coreset) < memory_size:
        if i < memory_size:
           agent.coreset.append((image[0], question[0], answer[0], weights[0], 1,agent.task_id + 1))
           # agent.coreset.append((image[0], question[0], list(answer), list(weights), list(n)))
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
        image, question, answer, weights, n = batch_data
        image, weights = image.to(device), weights.to(device)
        # **计算 loss**
        loss = model(image, question, answer, train=True, n=n, weights=weights, agent=agent)
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
        image_list = [item[0] for item in data_]
        question = [item[1] for item in data_]
        answer = [item[2] for item in data_]
        weights = [item[3] for item in data_]
        n = [item[4] for item in data_]

        image = torch.stack(image_list, dim=0)
        weights = torch.stack(weights, dim=0)

        # weights = weights.to(device)
        # **计算 loss**
        loss = model(image, question, answer, train=True, n=n, weights=weights, agent=agent)
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