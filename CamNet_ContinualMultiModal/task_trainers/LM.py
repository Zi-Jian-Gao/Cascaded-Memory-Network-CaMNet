import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
import os
from task_trainers.question_list import vqa_question_categories,pvqa_question_categories,gqa_question_categories
import random
def Create_LM(LM_path):
    # Use Accelerator to handle device management

    model_name = LM_path

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model = torch.nn.DataParallel(model)
    model.module.eval()

    return model, AutoTokenizer.from_pretrained(model_name, padding_side='left')



def generate_new_questions(model, tokenizer, questions, config,i):
    question_type_id = i % len(config['task_seq_name'])
    name = config['task_seq_name'][question_type_id]
    if 'pathvqa' in name:
        question_categories = pvqa_question_categories
    elif 'gqa' in name:
        question_categories = gqa_question_categories
    elif 'vqa' in name:
        question_categories = vqa_question_categories
    new_q_types = question_categories[name]
    new_q_types = [new_q_types[idx % len(new_q_types)] for idx in range(len(questions))]
    # 为每个问题生成新的 prompt
    prompts = [
        f"Generate a new question for me, replace the following question with the question which begins with '{new_q_type}', and make it grammatically correct: {question}. Only provide the new question that must differ from the old one."
        for question, new_q_type in zip(questions, new_q_types)
    ]

    inputs = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        inputs.append(text)

    model_device = next(model.module.parameters()).device
    model_inputs = tokenizer(inputs, return_tensors="pt", padding=True).to(model_device)

    generated_ids = model.module.generate(
        **model_inputs,
        max_new_tokens=512
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    responses = [response.split("\n", 1)[-1] for response in responses]

    # 清理多余的换行符、引号和其他不必要的符号
    cleaned_responses = []
    for response in responses:
        # 清除多余的换行符 \n 和双引号 "
        cleaned_response = response.strip()  # 去掉前后的空白字符
        cleaned_response = cleaned_response.replace('"', '')  # 去掉双引号
        cleaned_response = cleaned_response.replace('\n', ' ')  # 去掉多余的换行符
        cleaned_responses.append(cleaned_response)

    return responses, question_type_id + 1

def generate_new_concept(model, tokenizer, texts, config,i,agent):
    concept_type_id = i % len(config['task_seq_name'])
    name = config['task_seq_name'][concept_type_id]
    new_concpet = agent.concept_dict[name]
    if len(new_concpet) >= len(texts):
        random_concpet = random.sample(new_concpet, len(texts))
    else:
        random_concpet = random.choices(new_concpet, k=len(texts))  # 有放回采样

    # random_concpet = random.sample(new_concpet, len(texts))

    prompts = [
        f"Generate a new description for me, replace the following description with the description which begins with concept word '{new_concpet}', and make it as short as possible (no more than five words): {description}. Only provide the new description that must differ from the old one."
        for description, new_concpet in zip(texts, random_concpet)
    ]

    inputs = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        inputs.append(text)

    model_device = next(model.module.parameters()).device
    model_inputs = tokenizer(inputs, return_tensors="pt", padding=True).to(model_device)

    generated_ids = model.module.generate(
        **model_inputs,
        max_new_tokens=512
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    responses = [response.split("\n", 1)[-1] for response in responses]

    # 清理多余的换行符、引号和其他不必要的符号
    cleaned_responses = []
    for response in responses:
        # 清除多余的换行符 \n 和双引号 "
        cleaned_response = response.strip()  # 去掉前后的空白字符
        cleaned_response = cleaned_response.replace('"', '')  # 去掉双引号
        cleaned_response = cleaned_response.replace('\n', ' ')  # 去掉多余的换行符
        cleaned_responses.append(cleaned_response)

    return responses, concept_type_id + 1
