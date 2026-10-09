import os
import json
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
from data.utils import pre_question,pre_question_path,pre_answer

from torchvision.datasets.utils import download_url

class vqa_dataset(Dataset):
    def __init__(self, transform, json_file, split="train", config=None):
        self.split = split
        self.pathvqa_eos = '[SEP]'

        self.transform = transform
        if 'vqa_root' in config and config['vqa_root'] is not None:
            self.vqa_root = config['vqa_root']
            self.annotation = json.load(open(json_file, 'r'))
        elif 'gqa_root' in config and config['gqa_root'] is not None:
            self.gqa_root = config['gqa_root']
            self.annotation = json.load(open(json_file, 'r'))
        elif 'pathvqa_root' in config and config['pathvqa_root'] is not None:
            self.pathvqa_root = config['pathvqa_root']
            self.annotation = json.load(open(json_file, 'r'))
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        if ann['dataset']=='vqa':
            image_path = os.path.join(self.vqa_root,ann['image'][0])
        elif ann['dataset']=='gqa':
            image_path = os.path.join(self.gqa_root,ann['image'])
        elif ann['dataset']=='pathvqa':
            image_path = os.path.join(self.pathvqa_root,ann['image'])
            
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)          

        if ann['dataset']=='vqa':
            question = pre_question(ann['question'])
            answer_weight = {}
            for answer in ann['answer']:
                if answer in answer_weight.keys():
                    answer_weight[answer] += 1/len(ann['answer'])
                else:
                    answer_weight[answer] = 1/len(ann['answer'])
            answers = list(answer_weight.keys())
            weights = list(answer_weight.values())
        elif ann['dataset']=='gqa':
            question = pre_question(ann['question'])
            answers = [ann['answer']]
            weights = [0.2]
        elif ann['dataset']=='pathvqa':
            if self.split == 'train':
                question = pre_question_path(ann['question'],30)
            else:
                question = pre_question_path(ann['question'], 50)
            answers = ann['answer']
            answers = [pre_answer(answers)]
            if self.split == 'train':
                answers = [answer + self.pathvqa_eos for answer in answers]
            weights = [0.2]

        return image, question, answers, weights


class vqa_dataset_zsl(Dataset):
    def __init__(self, transform, json_file, split="train", config=None):
        self.split = split
        self.pathvqa_eos = '[SEP]'

        self.transform = transform
        if 'vqa_root' in config and config['vqa_root'] is not None:
            self.root = config['vqa_root']
            self.annotation = json.load(open(json_file, 'r'))
        elif 'gqa_root' in config and config['gqa_root'] is not None:
            self.root = config['gqa_root']
            self.annotation = json.load(open(json_file, 'r'))
        elif 'pathvqa_root' in config and config['pathvqa_root'] is not None:
            self.root = config['pathvqa_root']
            self.annotation = json.load(open(json_file, 'r'))

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        ann = self.annotation[index]

        if ann['dataset'] == 'vqa':
            image_path = os.path.join(self.root, ann['image'][0])
        elif ann['dataset'] == 'gqa':
            image_path = os.path.join(self.root, ann['image'])
        elif ann['dataset']=='pathvqa':
            image_path = os.path.join(self.root,ann['image'])

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        if ann['dataset'] == 'vqa':
            question = pre_question(ann['question'])
            answer_weight = {}
            for answer in ann['answer']:
                if answer in answer_weight.keys():
                    answer_weight[answer] += 1 / len(ann['answer'])
                else:
                    answer_weight[answer] = 1 / len(ann['answer'])

            answers = list(answer_weight.keys())
            weights = list(answer_weight.values())
        elif ann['dataset'] == 'gqa':
            question = pre_question(ann['question'])
            answers = [ann['answer']]
            weights = [0.2]
        elif ann['dataset']=='pathvqa':
            if self.split == 'train':
                question = pre_question_path(ann['question'],30)
            else:
                question = pre_question_path(ann['question'], 50)
            answers = ann['answer']
            answers = [pre_answer(answers)]
            if self.split == 'train':
                answers = [answer + self.pathvqa_eos for answer in answers]
            weights = [0.2]

        wild_id = int(ann['wild_id']) if 'wild_id' in ann and ann['wild_id'] else -1

        return image, question, answers, weights, wild_id

class vqa_dataset_fsl(Dataset):
    def __init__(self, transform, json_file, split="train", config=None):
        self.split = split
        self.pathvqa_eos = '[SEP]'

        self.transform = transform
        if 'vqa_root' in config and config['vqa_root'] is not None:
            self.root = config['vqa_root']
            self.annotation = json.load(open(json_file, 'r'))
        elif 'vg_root' in config and config['vg_root'] is not None:
            self.root = config['vg_root']
            self.annotation = json.load(open(json_file, 'r'))
        elif 'pathvqa_root' in config and config['pathvqa_root'] is not None:
            self.root = config['pathvqa_root']
            self.annotation = json.load(open(json_file, 'r'))

    def __len__(self):
        return len(self.annotation)

    def __getitem__(self, index):

        ann = self.annotation[index]

        if ann['dataset'] == 'vqa':
            image_path = os.path.join(self.root, ann['image'][0])
        elif ann['dataset'] == 'vg':
            image_path = os.path.join(self.root, ann['image'])
        elif ann['dataset']=='pathvqa':
            image_path = os.path.join(self.root,ann['image'])

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        if ann['dataset'] == 'vqa':
            question = pre_question(ann['question'])
            answer_weight = {}
            for answer in ann['answer']:
                if answer in answer_weight.keys():
                    answer_weight[answer] += 1 / len(ann['answer'])
                else:
                    answer_weight[answer] = 1 / len(ann['answer'])

            answers = list(answer_weight.keys())
            weights = list(answer_weight.values())
        elif ann['dataset'] == 'vg':
            question = pre_question(ann['question'])
            answers = [ann['answer']]
            weights = [0.2]
        elif ann['dataset']=='pathvqa':
            if self.split == 'train':
                question = pre_question_path(ann['question'],30)
            else:
                question = pre_question_path(ann['question'], 50)
            answers = ann['answer']
            answers = [pre_answer(answers)]
            if self.split == 'train':
                answers = [answer + self.pathvqa_eos for answer in answers]
            weights = [0.2]

        return image, question, answers, weights



def vqa_collate_fn_zsl(batch):
    image_list, question_list, answer_list, weight_list, n,wild_id_list = [], [], [], [], [], []
    for image, question, answer, weights,wild_id in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights
        answer_list += answer
        n.append(len(answer))
        wild_id_list.append(wild_id)
    return torch.stack(image_list,dim=0), question_list, answer_list, torch.Tensor(weight_list), n, wild_id_list

def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights       
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list,dim=0), question_list, answer_list, torch.Tensor(weight_list), n        