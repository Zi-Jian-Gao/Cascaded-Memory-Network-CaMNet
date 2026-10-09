from models.med import BertConfig, BertModel, BertLMHeadModel
from models.blip import create_vit, init_tokenizer, load_checkpoint

import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizer
import numpy as np
from transformers import GenerationConfig
class BLIP_VQA(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 480,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,
                 agent = None
                 ):
        """
        Args:
            med_config (str): path for the mixture of encoder-decoder model's configuration file
            image_size (int): input image size
            vit (str): model size of vision transformer
        """               
        super().__init__()
        
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer, drop_path_rate=0.1, agent=agent)
        self.tokenizer = init_tokenizer()

        encoder_config = BertConfig.from_json_file(med_config)
        encoder_config.encoder_width = vision_width
        self.text_encoder = BertModel(config=encoder_config, add_pooling_layer=False, agent=agent)
        
        decoder_config = BertConfig.from_json_file(med_config)
        self.text_decoder = BertLMHeadModel(config=decoder_config, agent=agent)


    def forward(self, image, question, answer=None, n=None, weights=None, train=True, inference='generate', k_test=128, train_zsl=False, agent = None,wild_id = None,coreset_input=False,LM_replay=False):
        
        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        
        question = self.tokenizer(question, padding='longest', truncation=True, max_length=35, 
                                  return_tensors="pt").to(image.device) 
        question.input_ids[:,0] = self.tokenizer.enc_token_id

        if train :
            if not train_zsl:
                '''
                n: number of answers for each question
                weights: weight for each answer
                '''
                answer = self.tokenizer(answer, padding='longest', return_tensors="pt").to(image.device)
                answer.input_ids[:,0] = self.tokenizer.bos_token_id
                answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.tokenizer.pad_token_id, -100)

                question_output = self.text_encoder(question.input_ids,
                                                    attention_mask = question.attention_mask,
                                                    encoder_hidden_states = image_embeds,
                                                    encoder_attention_mask = image_atts,
                                                    return_dict = True)

                question_states = []
                question_atts = []
                for b, i in enumerate(n):
                    question_states += [question_output.last_hidden_state[b]]*i
                    question_atts += [question.attention_mask[b]]*i
                question_states = torch.stack(question_states,0)
                question_atts = torch.stack(question_atts,0)
                if not LM_replay:
                    answer_output = self.text_decoder(answer.input_ids,
                                                      attention_mask = answer.attention_mask,
                                                      encoder_hidden_states = question_states,
                                                      encoder_attention_mask = question_atts,
                                                      labels = answer_targets,
                                                      return_dict = True,
                                                      reduction = 'none'
                                                     )
                    loss = weights * answer_output.loss
                    loss = loss.sum()/image.size(0)
                    prediction = answer_output.logits
                    # we are doing next-token prediction; shift prediction scores and input ids by one
                    prediction = prediction[:, :-1, :].contiguous()
                else:
                    answer_output = self.text_decoder(answer.input_ids,
                                                      attention_mask=answer.attention_mask,
                                                      encoder_hidden_states=question_states,
                                                      encoder_attention_mask=question_atts,
                                                      labels=None,
                                                      return_dict=True,
                                                      reduction='none'
                                                      )
                    prediction = answer_output.logits
                    # we are doing next-token prediction; shift prediction scores and input ids by one
                    prediction = prediction[:, :-1, :].contiguous()

                alpha = agent.args.loss_alpha
                if agent.train_distill_type is not None:
                    if agent.train_distill_type == 'adv_text':
                        assert agent.args.freeze_text_emb  # without this it is not clear how we transfer text in soft form between task models
                        alpha = agent.args.loss_alpha
                        if agent.args.auto_scale_alpha and (agent.task_id > 0):
                            alpha /= agent.task_id
                        with torch.no_grad():
                            base_text_emb = self.text_encoder._embed_only(question.input_ids).detach()
                        for task_id in range(agent.task_id):
                            if agent.args.adv_last_only:
                                if task_id < (agent.task_id - agent.args.adv_num_last):
                                    continue
                            step_sz = agent.args.adv_step_sz
                            num_steps = agent.args.num_adv_iters

                            adv_text_emb = base_text_emb.detach()
                            for iStep in range(num_steps):
                                adv_text_emb.requires_grad = True
                                answer_output_ = self.get_task_feats_cons(image, adv_text_emb, question.attention_mask,
                                                                     task_id, agent, answer, answer_targets, n,
                                                                     no_detach=True, text_is_emb=True)
                                adv_pred = answer_output_.logits
                                if iStep == 0:
                                    orig_prev_pred = adv_pred.detach()
                                if iStep < (num_steps - 1):
                                    adv_loss = answer_output_.loss
                                    adv_loss_scalar = adv_loss.mean()
                                    grad = torch.autograd.grad(adv_loss_scalar, adv_text_emb, create_graph=False)
                                    adv_text_emb = adv_text_emb + step_sz * grad[0].sign()
                                adv_text_emb = adv_text_emb.detach()
                            adv_pred = adv_pred.detach()

                            cur_adv_answer_output = self.get_task_feats_cons(image, adv_text_emb, question.attention_mask,
                                                                        1e7, agent, answer, answer_targets, n,
                                                                        no_detach=True, text_is_emb=True)
                            cur_adv_pred = cur_adv_answer_output.logits
                            _prediction = answer_output.logits

                            kl_loss_1 = _KL_loss(orig_prev_pred, adv_pred, 2)
                            kl_loss_2 = _KL_loss(_prediction,cur_adv_pred, 2)
                            loss += alpha * torch.abs(kl_loss_1 - kl_loss_2)
                        return loss
                if agent.ema and agent.task_id!=0 and coreset_input:
                    with torch.no_grad():
                        fuse_type = agent.fuse_type

                        agent.fuse_type = 'ema'

                        image_embeds = self.visual_encoder(image)
                        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

                        question_output = self.text_encoder(question.input_ids,
                                                            attention_mask=question.attention_mask,
                                                            encoder_hidden_states=image_embeds,
                                                            encoder_attention_mask=image_atts,
                                                            return_dict=True)

                        question_states = []
                        question_atts = []
                        for b, i in enumerate(n):
                            question_states += [question_output.last_hidden_state[b]] * i
                            question_atts += [question.attention_mask[b]] * i
                        question_states = torch.stack(question_states, 0)
                        question_atts = torch.stack(question_atts, 0)

                        answer_output = self.text_decoder(answer.input_ids,
                                                          attention_mask=answer.attention_mask,
                                                          encoder_hidden_states=question_states,
                                                          encoder_attention_mask=question_atts,
                                                          labels=None,
                                                          return_dict=True,
                                                          reduction='none'
                                                          )

                        slow_prediction = answer_output.logits
                        # we are doing next-token prediction; shift prediction scores and input ids by one
                        slow_prediction = slow_prediction[:, :-1, :].contiguous()
                        agent.fuse_type = fuse_type
                    if agent.LM:
                        loss = alpha * _KL_loss(prediction, slow_prediction, 1)
                    else:
                        loss = loss + alpha * _KL_loss(prediction, slow_prediction, 1)

                elif agent.cls and agent.task_id!=0 and coreset_input:
                    with torch.no_grad():
                        fuse_type = agent.fuse_type
                        agent.fuse_type = 'slow'
                        image_embeds = self.visual_encoder(image)
                        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

                        question_output = self.text_encoder(question.input_ids,
                                                            attention_mask=question.attention_mask,
                                                            encoder_hidden_states=image_embeds,
                                                            encoder_attention_mask=image_atts,
                                                            return_dict=True)

                        question_states = []
                        question_atts = []
                        for b, i in enumerate(n):
                            question_states += [question_output.last_hidden_state[b]] * i
                            question_atts += [question.attention_mask[b]] * i
                        question_states = torch.stack(question_states, 0)
                        question_atts = torch.stack(question_atts, 0)

                        answer_output = self.text_decoder(answer.input_ids,
                                                          attention_mask=answer.attention_mask,
                                                          encoder_hidden_states=question_states,
                                                          encoder_attention_mask=question_atts,
                                                          labels=None,
                                                          return_dict=True,
                                                          reduction='none'
                                                          )

                        slow_prediction = answer_output.logits
                        # we are doing next-token prediction; shift prediction scores and input ids by one
                        slow_prediction = slow_prediction[:, :-1, :].contiguous()

                        agent.fuse_type = 'fast'

                        image_embeds = self.visual_encoder(image)
                        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

                        question_output = self.text_encoder(question.input_ids,
                                                            attention_mask=question.attention_mask,
                                                            encoder_hidden_states=image_embeds,
                                                            encoder_attention_mask=image_atts,
                                                            return_dict=True)

                        question_states = []
                        question_atts = []
                        for b, i in enumerate(n):
                            question_states += [question_output.last_hidden_state[b]] * i
                            question_atts += [question.attention_mask[b]] * i
                        question_states = torch.stack(question_states, 0)
                        question_atts = torch.stack(question_atts, 0)

                        answer_output = self.text_decoder(answer.input_ids,
                                                          attention_mask=answer.attention_mask,
                                                          encoder_hidden_states=question_states,
                                                          encoder_attention_mask=question_atts,
                                                          labels=None,
                                                          return_dict=True,
                                                          reduction='none'
                                                          )

                        fast_prediction = answer_output.logits
                        # we are doing next-token prediction; shift prediction scores and input ids by one
                        fast_prediction = fast_prediction[:, :-1, :].contiguous()

                        agent.fuse_type = fuse_type

                    if agent.LM:
                        loss = alpha * _KL_loss(prediction, slow_prediction, 1) + alpha * _KL_loss(prediction, fast_prediction, 1)
                    else:
                        loss = loss + alpha * _KL_loss(prediction, slow_prediction, 1) + alpha *  _KL_loss(prediction, fast_prediction, 1)

                elif agent.LwF and agent.task_id!=0 and (not agent.LwF_ER or coreset_input) and (not agent.LM or coreset_input):
                    with torch.no_grad():
                        fuse_type = agent.fuse_type
                        agent.fuse_type = 'LwF'
                        image_embeds = self.visual_encoder(image)
                        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

                        question_output = self.text_encoder(question.input_ids,
                                                            attention_mask=question.attention_mask,
                                                            encoder_hidden_states=image_embeds,
                                                            encoder_attention_mask=image_atts,
                                                            return_dict=True)

                        question_states = []
                        question_atts = []
                        for b, i in enumerate(n):
                            question_states += [question_output.last_hidden_state[b]] * i
                            question_atts += [question.attention_mask[b]] * i
                        question_states = torch.stack(question_states, 0)
                        question_atts = torch.stack(question_atts, 0)

                        answer_output = self.text_decoder(answer.input_ids,
                                                          attention_mask=answer.attention_mask,
                                                          encoder_hidden_states=question_states,
                                                          encoder_attention_mask=question_atts,
                                                          labels=None,
                                                          return_dict=True,
                                                          reduction='none'
                                                          )

                        past_prediction = answer_output.logits
                        # we are doing next-token prediction; shift prediction scores and input ids by one
                        past_prediction = past_prediction[:, :-1, :].contiguous()
                        agent.fuse_type = fuse_type
                    if agent.LM:
                        loss = alpha * _KL_loss(prediction, past_prediction, 1)
                    else:
                        loss = loss + alpha * _KL_loss(prediction, past_prediction, 1)
                return loss

            else:
                answer = self.tokenizer(answer, padding='longest', return_tensors="pt").to(image.device)
                answer.input_ids[:, 0] = self.tokenizer.bos_token_id
                # answer_targets = answer.input_ids.masked_fill(answer.input_ids == self.tokenizer.pad_token_id, -100)

                alpha = agent.args.loss_alpha

                question_output = self.text_encoder(question.input_ids,
                                                    attention_mask=question.attention_mask,
                                                    encoder_hidden_states=image_embeds,
                                                    encoder_attention_mask=image_atts,
                                                    return_dict=True)

                question_states = []
                question_atts = []
                for b, i in enumerate(n):
                    question_states += [question_output.last_hidden_state[b]] * i
                    question_atts += [question.attention_mask[b]] * i
                question_states = torch.stack(question_states, 0)
                question_atts = torch.stack(question_atts, 0)

                answer_output = self.text_decoder(answer.input_ids,
                                                  attention_mask=answer.attention_mask,
                                                  encoder_hidden_states=question_states,
                                                  encoder_attention_mask=question_atts,
                                                  labels=None,
                                                  return_dict=True,
                                                  reduction='none'
                                                  )

                prediction = answer_output.logits

                # we are doing next-token prediction; shift prediction scores and input ids by one
                prediction = prediction[:, :-1, :].contiguous()
                if agent.train_distill_type == 'ema-zsl-single':
                    fuse_type = agent.fuse_type
                    agent.fuse_type = 'ema'
                    with torch.no_grad():
                        image_embeds = self.visual_encoder(image)
                        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
                        question_output = self.text_encoder(question.input_ids,
                                                            attention_mask=question.attention_mask,
                                                            encoder_hidden_states=image_embeds,
                                                            encoder_attention_mask=image_atts,
                                                            return_dict=True)

                        question_states = []
                        question_atts = []
                        for b, i in enumerate(n):
                            question_states += [question_output.last_hidden_state[b]] * i
                            question_atts += [question.attention_mask[b]] * i
                        question_states = torch.stack(question_states, 0)
                        question_atts = torch.stack(question_atts, 0)

                        answer_output = self.text_decoder(answer.input_ids,
                                                          attention_mask=answer.attention_mask,
                                                          encoder_hidden_states=question_states,
                                                          encoder_attention_mask=question_atts,
                                                          labels=None,
                                                          return_dict=True,
                                                          reduction='none'
                                                          )

                        zsl_prediction = answer_output.logits
                        # we are doing next-token prediction; shift prediction scores and input ids by one
                        zsl_prediction = zsl_prediction[:, :-1, :].contiguous()
                        agent.fuse_type = fuse_type

                elif agent.train_distill_type == 'multi-zsl-single':
                    with torch.no_grad():
                        fuse_type = agent.fuse_type
                        agent.fuse_type = 'multi-zsl-single'

                        image_embeds = self.visual_encoder(image,wild_id =wild_id)
                        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

                        question_output = self.text_encoder(question.input_ids,
                                                            attention_mask=question.attention_mask,
                                                            encoder_hidden_states=image_embeds,
                                                            encoder_attention_mask=image_atts,
                                                            return_dict=True,wild_id =wild_id)

                        question_states = []
                        question_atts = []
                        for b, i in enumerate(n):
                            question_states += [question_output.last_hidden_state[b]] * i
                            question_atts += [question.attention_mask[b]] * i
                        question_states = torch.stack(question_states, 0)
                        question_atts = torch.stack(question_atts, 0)

                        answer_output = self.text_decoder(answer.input_ids,
                                                          attention_mask=answer.attention_mask,
                                                          encoder_hidden_states=question_states,
                                                          encoder_attention_mask=question_atts,
                                                          labels=None,
                                                          return_dict=True,
                                                          reduction='none',wild_id =wild_id
                                                          )

                        zsl_prediction = answer_output.logits
                        # we are doing next-token prediction; shift prediction scores and input ids by one
                        zsl_prediction = zsl_prediction[:, :-1, :].contiguous()

                    if agent.slow:
                        with torch.no_grad():
                            image_embeds = self.visual_encoder(image,wild_id =[-1])
                            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

                            question_output = self.text_encoder(question.input_ids,
                                                                attention_mask=question.attention_mask,
                                                                encoder_hidden_states=image_embeds,
                                                                encoder_attention_mask=image_atts,
                                                                return_dict=True,wild_id =[-1])

                            question_states = []
                            question_atts = []
                            for b, i in enumerate(n):
                                question_states += [question_output.last_hidden_state[b]] * i
                                question_atts += [question.attention_mask[b]] * i
                            question_states = torch.stack(question_states, 0)
                            question_atts = torch.stack(question_atts, 0)

                            answer_output = self.text_decoder(answer.input_ids,
                                                              attention_mask=answer.attention_mask,
                                                              encoder_hidden_states=question_states,
                                                              encoder_attention_mask=question_atts,
                                                              labels=None,
                                                              return_dict=True,
                                                              reduction='none',wild_id =[-1]
                                                              )

                            cur_zsl_prediction = answer_output.logits
                            cur_zsl_prediction = cur_zsl_prediction[:, :-1, :].contiguous()

                agent.fuse_type = fuse_type


                if agent.slow and agent.train_distill_type == 'multi-zsl-single':
                    loss = alpha * _KL_loss(prediction, zsl_prediction, 1) + alpha * _KL_loss(prediction, cur_zsl_prediction, 1)
                else:
                    loss = alpha * _KL_loss(prediction,zsl_prediction,1)

                return loss

        else:
            question_output = self.text_encoder(question.input_ids, 
                                                attention_mask = question.attention_mask, 
                                                encoder_hidden_states = image_embeds,
                                                encoder_attention_mask = image_atts,                                    
                                                return_dict = True)

            if inference=='generate':
                num_beams = 3
                question_states = question_output.last_hidden_state
                question_atts = torch.ones(question_states.size()[:-1],dtype=torch.long).to(question_states.device)
                model_kwargs = {"encoder_hidden_states": question_states, "encoder_attention_mask":question_atts}

                bos_ids = torch.full((image.size(0), 1), fill_value=self.tokenizer.bos_token_id, device=image.device)

                gen_config = GenerationConfig(
                    max_length=10,
                    min_length=1,
                    num_beams=num_beams,
                    do_sample=False,
                    early_stopping=True,
                    length_penalty=0.8,
                    eos_token_id=self.tokenizer.sep_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    output_hidden_states=False,  # 避免意外返回结构
                )

                outputs = self.text_decoder.generate(
                    input_ids=bos_ids,
                    generation_config=gen_config,  # 关键点：强制替换 config
                    **model_kwargs
                )
                answers = []
                for output in outputs:
                    answer = self.tokenizer.decode(output, skip_special_tokens=True)
                    answers.append(answer)
                return answers

 
                
                
    def rank_answer(self, question_states, question_atts, answer_ids, answer_atts, k):
        
        num_ques = question_states.size(0)
        start_ids = answer_ids[0,0].repeat(num_ques,1) # bos token
        
        start_output = self.text_decoder(start_ids, 
                                         encoder_hidden_states = question_states,
                                         encoder_attention_mask = question_atts,                                      
                                         return_dict = True,
                                         reduction = 'none')              
        logits = start_output.logits[:,0,:] # first token's logit
        
        # topk_probs: top-k probability 
        # topk_ids: [num_question, k]        
        answer_first_token = answer_ids[:,1]
        prob_first_token = F.softmax(logits,dim=1).index_select(dim=1, index=answer_first_token) 
        topk_probs, topk_ids = prob_first_token.topk(k,dim=1) 
        
        # answer input: [num_question*k, answer_len]                 
        input_ids = []
        input_atts = []
        for b, topk_id in enumerate(topk_ids):
            input_ids.append(answer_ids.index_select(dim=0, index=topk_id))
            input_atts.append(answer_atts.index_select(dim=0, index=topk_id))
        input_ids = torch.cat(input_ids,dim=0)  
        input_atts = torch.cat(input_atts,dim=0)  

        targets_ids = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)

        # repeat encoder's output for top-k answers
        question_states = tile(question_states, 0, k)
        question_atts = tile(question_atts, 0, k)
        
        output = self.text_decoder(input_ids, 
                                   attention_mask = input_atts, 
                                   encoder_hidden_states = question_states,
                                   encoder_attention_mask = question_atts,     
                                   labels = targets_ids,
                                   return_dict = True, 
                                   reduction = 'none')   
        
        log_probs_sum = -output.loss
        log_probs_sum = log_probs_sum.view(num_ques,k)

        max_topk_ids = log_probs_sum.argmax(dim=1) 
        max_ids = topk_ids[max_topk_ids>=0,max_topk_ids]

        return max_ids

    def _get_task_feats_(self, image, text_input_ids, text_attention_mask,answer,answer_targets,n, no_detach=False, text_is_emb=False):
        image_embeds_q = self.visual_encoder(image)
        if not no_detach:
            image_embeds_q = image_embeds_q.detach()
        image_atts_q = torch.ones(image_embeds_q.size()[:-1], dtype=torch.long).to(image.device)
        question_output = self.text_encoder(text_input_ids if (not text_is_emb) else None,
                                          encoder_embeds=(text_input_ids if text_is_emb else None),
                                          attention_mask=text_attention_mask,
                                          encoder_hidden_states=image_embeds_q,
                                          encoder_attention_mask=image_atts_q,
                                          return_dict=True,
                                          )
        if not no_detach:
            question_output = question_output.detach()
        image_embeds_q = image_embeds_q.mean(dim=1)
        if not no_detach:
            image_embeds_q = image_embeds_q.detach()

        question_states = []
        question_atts = []
        for b, i in enumerate(n):
            question_states += [question_output.last_hidden_state[b]] * i
            question_atts += [text_attention_mask.attention_mask[b]] * i
        question_states = torch.stack(question_states, 0)
        question_atts = torch.stack(question_atts, 0)

        answer_output = self.text_decoder(answer.input_ids,
                                          attention_mask=answer.attention_mask,
                                          encoder_hidden_states=question_states,
                                          encoder_attention_mask=question_atts,
                                          labels=None,
                                          return_dict=True,
                                          reduction='none',
                                          )

        return answer_output




    def get_task_feats(self, image, text_input_ids, text_attention_mask, task_id, agent,answer,answer_targets,n,  no_detach=False, text_is_emb=False):
        agent.prep_model4task(task_id, force=True)
        if not no_detach:
            with torch.no_grad():
                text_embeds_q = self._get_task_feats_(image, text_input_ids, text_attention_mask, answer,answer_targets, n, no_detach, text_is_emb)
        else:
            text_embeds_q = self._get_task_feats_(image, text_input_ids, text_attention_mask, answer,answer_targets, n, no_detach, text_is_emb)
        agent.prep_model4task(-1)
        return text_embeds_q

    def get_task_feats_cons(self, image, text_input_ids, text_attention_mask, task_id, agent, answer, answer_targets, n,
                            no_detach=False, text_is_emb=False):
        agent.prep_model4task(task_id, force=True)
        if not no_detach:
            with torch.no_grad():
                text_embeds_q = self._get_task_feats_cons(image, text_input_ids, text_attention_mask, answer,
                                                      answer_targets, n,
                                                      no_detach, text_is_emb)
        else:
            text_embeds_q = self._get_task_feats_cons(image, text_input_ids, text_attention_mask, answer, answer_targets, n,
                                                  no_detach, text_is_emb)
        agent.prep_model4task(-1)
        return text_embeds_q

    def _get_task_feats_cons(self, image, text_input_ids, text_attention_mask, answer, answer_targets, n,
                             no_detach=False,
                             text_is_emb=False):
        image_embeds_q = self.visual_encoder(image)
        if not no_detach:
            image_embeds_q = image_embeds_q.detach()
        image_atts_q = torch.ones(image_embeds_q.size()[:-1], dtype=torch.long).to(image.device)
        question_output = self.text_encoder(text_input_ids if (not text_is_emb) else None,
                                            encoder_embeds=(text_input_ids if text_is_emb else None),
                                            attention_mask=text_attention_mask,
                                            encoder_hidden_states=image_embeds_q,
                                            encoder_attention_mask=image_atts_q,
                                            return_dict=True,
                                            )
        if not no_detach:
            question_output = question_output.detach()
        image_embeds_q = image_embeds_q.mean(dim=1)
        if not no_detach:
            image_embeds_q = image_embeds_q.detach()

        question_states = []
        question_atts = []
        for b, i in enumerate(n):
            question_states += [question_output.last_hidden_state[b]] * i
            #modify here
            question_atts += [text_attention_mask[b]] * i
        question_states = torch.stack(question_states, 0)
        question_atts = torch.stack(question_atts, 0)

        answer_output = self.text_decoder(answer.input_ids,
                                          attention_mask=answer.attention_mask,
                                          encoder_hidden_states=question_states,
                                          encoder_attention_mask=question_atts,
                                          labels=answer_targets,
                                          return_dict=True,
                                          reduction='none',
                                          )

        return answer_output
    
def blip_vqa(pretrained='',**kwargs):
    model = BLIP_VQA(**kwargs)
    head_not_loaded = True
    if pretrained:
        model,msg = load_checkpoint(model,pretrained)
#         assert(len(msg.missing_keys)==0)
        print("missing keys:")
        print(msg.missing_keys)
        head_not_loaded = False
        for k in msg.missing_keys:
            if 'text_decoder.cls' in k:
                head_not_loaded = True

    return model, head_not_loaded
    # return model


def tile(x, dim, n_tile):
    init_dim = x.size(dim)
    repeat_idx = [1] * x.dim()
    repeat_idx[dim] = n_tile
    x = x.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(x, dim, order_index.to(x.device))    
        
def _KL_loss(pred, soft, T):

#    pred = torch.log_softmax(pred / T, dim=1)
#    soft = torch.softmax(soft / T, dim=1)
    loss = nn.KLDivLoss(reduction='batchmean')(torch.log_softmax(pred / T, dim=-1),
                                                       torch.softmax(soft / T, dim=-1)) * T * T
    return loss

