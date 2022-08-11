import copy
import torch
import torch.nn as nn
import numpy as np

from modeling_t5 import VLT5

class VLT5DialPre(VLT5):
    
    def __init__(self, config):
        super().__init__(config)
        self.losses = self.config.dial_losses.split(',')

    def train_step(self, batch):
        device = next(self.parameters()).device
        input_ids = batch['input_ids'].to(device)
        B = len(input_ids)
        labels = batch['target_ids'].to(device)
        if not self.config.just_text_features:
            feat_dim = batch['vis_feats'].size(3)
            n_boxes = batch['vis_feats'].size(2)
            context_images_amount = batch['vis_feats'].size(1)
            
            vis_feats = batch['vis_feats'].to(device).view(B, context_images_amount*n_boxes, feat_dim)
            vis_pos = batch['boxes'].to(device).view(B, context_images_amount*n_boxes, 4)
            vis_attention_mask = batch['vis_attention_mask'].to(device).view(B, context_images_amount*n_boxes)
            
            img_order_ids = batch['img_order_ids'].to(device).view(B, context_images_amount*n_boxes)

            if self.config.match_text_image:
                img_order_ids = batch['img_order_encoded'].to(device).view(B, context_images_amount*n_boxes)

            if not self.config.multi_image_pretrain:
                obj_order_ids = torch.arange(n_boxes, dtype=torch.long, device=device)
                obj_order_ids = torch.tensor(self.tokenizer.encode([f'<vis_extra_id_{index}>' for index in obj_order_ids], add_special_tokens=False))
                obj_order_ids = obj_order_ids.view(1, 1, n_boxes).expand(B, context_images_amount, -1).contiguous().view(B, context_images_amount*n_boxes).to(device)
            else:
                obj_order_ids = batch['obj_order_ids'].to(device)

            output = self(input_ids=input_ids,
                        vis_inputs=(vis_feats, vis_pos, img_order_ids, obj_order_ids),
                        vis_attention_mask=vis_attention_mask,
                        labels=labels,
                        return_dict=True)
        else:
            output = self(input_ids=input_ids,
                        labels=labels,
                        return_dict=True)

        # Don't take padding tokens into account for loss
        lm_mask = (labels != -100).float()
        B, L = labels.size()

        loss = output['loss']
        loss = loss.view(B, L) * lm_mask
        loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)  # B
        task_counts = {task: 0 for task in self.losses}
        task_loss = {task: 0 for task in self.losses}

        results = {}

        results['loss'] = (loss).mean()
        results['total_loss'] = loss.detach().sum()
        results['total_loss_count'] = len(loss)

        task_counts = {task: 0 for task in self.losses}
        task_loss = {task: 0 for task in self.losses}

        for _loss, task in zip(loss.detach(), batch['task']):
            task_loss[task] += _loss
            task_counts[task] += 1

        for task in self.losses:
            if task_counts[task] > 0:
                results[f'{task}_loss'] = task_loss[task]
                results[f'{task}_loss_count'] = task_counts[task]

        return results

    @torch.no_grad()
    def valid_step(self, batch):
        self.eval()

        device = next(self.parameters()).device
        input_ids = batch['input_ids'].to(device)
        B = len(input_ids)
        
        labels = batch['target_ids'].to(device)
       
        if not self.config.just_text_features:
            feat_dim = batch['vis_feats'].size(3)
            n_boxes = batch['vis_feats'].size(2)
            context_images_amount = batch['vis_feats'].size(1)
            
            vis_feats = batch['vis_feats'].to(device).view(B, context_images_amount*n_boxes, feat_dim)
            vis_pos = batch['boxes'].to(device).view(B, context_images_amount*n_boxes, 4)
            vis_attention_mask = batch['vis_attention_mask'].to(device).view(B, context_images_amount*n_boxes)

            img_order_ids = batch['img_order_ids'].to(device).view(B, context_images_amount*n_boxes)

            if self.config.match_text_image:
                img_order_ids = batch['img_order_encoded'].to(device).view(B, context_images_amount*n_boxes)
            
            if not self.config.multi_image_pretrain:
                obj_order_ids = torch.arange(n_boxes, dtype=torch.long, device=device)
                obj_order_ids = torch.tensor(self.tokenizer.encode([f'<vis_extra_id_{index}>' for index in obj_order_ids], add_special_tokens=False))
                obj_order_ids = obj_order_ids.view(1, 1, n_boxes).expand(B, context_images_amount, -1).contiguous().view(B, context_images_amount*n_boxes).to(device)
            else:
                obj_order_ids = batch['obj_order_ids'].to(device).view(B, context_images_amount*n_boxes)   

            output = self(input_ids=input_ids,
                        vis_inputs=(vis_feats, vis_pos, img_order_ids, obj_order_ids),
                        vis_attention_mask=vis_attention_mask,
                        labels=labels,
                        return_dict=True)
        else:
            output = self(input_ids=input_ids,
                        labels=labels,
                        return_dict=True)

        # Don't take padding tokens into account for loss
        lm_mask = (labels != -100).float()
        B, L = labels.size()

        loss = output['loss']
        loss = loss.view(B, L) * lm_mask
        loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)  # B

        results = {}

        results['loss'] = (loss).mean()
        results['total_loss'] = loss.detach().sum()
        results['total_loss_count'] = len(loss)

        task_counts = {task: 0 for task in self.losses}
        task_loss = {task: 0 for task in self.losses}

        for _loss, task in zip(loss.detach(), batch['task']):
            task_loss[task] += _loss
            task_counts[task] += 1

        for task in self.losses:
            if task_counts[task] > 0:
                # result[f'{task}_loss'] = task_loss[task] / task_counts[task]
                results[f'{task}_loss'] = task_loss[task]
                results[f'{task}_loss_count'] = task_counts[task]
            # else:
            #     result[f'{task}_loss'] = torch.zeros_like(loss)

        return results

    @torch.no_grad()
    def test_step(self, batch, **kwargs):
        self.eval()

        device = next(self.parameters()).device
        input_ids = batch['input_ids'].to(device)
        B = len(input_ids)
        decoder_input_ids = torch.ones(B, 1, dtype=torch.long, device=device) * self.config.decoder_start_token_id
               
        if not self.config.just_text_features:
            feat_dim = batch['vis_feats'].size(3)
            n_boxes = batch['vis_feats'].size(2)
            context_images_amount = batch['vis_feats'].size(1)
            
            vis_feats = batch['vis_feats'].to(device).view(B, context_images_amount*n_boxes, feat_dim)
            vis_pos = batch['boxes'].to(device).view(B, context_images_amount*n_boxes, 4)
            vis_attention_mask = batch['vis_attention_mask'].to(device).view(B, context_images_amount*n_boxes)

            img_order_ids = batch['img_order_ids'].to(device).view(B, context_images_amount*n_boxes)

            if self.config.match_text_image:
                img_order_ids = batch['img_order_encoded'].to(device).view(B, context_images_amount*n_boxes)
           
            if not self.config.multi_image_pretrain:
                obj_order_ids = torch.arange(n_boxes, dtype=torch.long, device=device)
                obj_order_ids = torch.tensor(self.tokenizer.encode([f'<vis_extra_id_{index}>' for index in obj_order_ids], add_special_tokens=False))
                obj_order_ids = obj_order_ids.view(1, 1, n_boxes).expand(B, context_images_amount, -1).contiguous().view(B, context_images_amount*n_boxes).to(device)
            else:
                obj_order_ids = batch['obj_order_ids'].to(device).view(B, context_images_amount*n_boxes)

            output = self.generate(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos, img_order_ids, obj_order_ids),
                vis_attention_mask=vis_attention_mask,
                decoder_input_ids=decoder_input_ids,
                **kwargs
            )
        else:
            output = self.generate(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                **kwargs
            )

        processed = []
        for element in output.cpu().tolist():
            try:
                processed.append(element[element.index(self.tokenizer.pad_token_id)+1:element[1:].index(self.tokenizer.eos_token_id)+1])
            except:
                continue
            
        generated_sents = self.tokenizer.batch_decode(processed)
        
        return {'token_ids': processed,
                'pred': generated_sents}        
        
from modeling_bart import VLBart

class VLBartDialPre(VLBart):

    def __init__(self, config):
        super().__init__(config)

    def train_step(self, batch):

        device = next(self.parameters()).device
        input_ids = batch['input_ids'].to(device)
        B = len(input_ids)
        labels = batch['target_ids'].to(device)
       
        if not self.config.just_text_features:
            feat_dim = batch['vis_feats'].size(3)
            n_boxes = batch['vis_feats'].size(2)
            context_images_amount = batch['vis_feats'].size(1)
            
            vis_feats = batch['vis_feats'].to(device).view(B, context_images_amount*n_boxes, feat_dim)
            vis_pos = batch['boxes'].to(device).view(B, context_images_amount*n_boxes, 4)
            vis_attention_mask = batch['vis_attention_mask'].to(device).view(B, context_images_amount*n_boxes)

            img_order_ids = batch['img_order_ids'].to(device).view(B, context_images_amount*n_boxes)

            if self.config.match_text_image:
                img_order_ids = batch['img_order_encoded'].to(device).view(B, context_images_amount*n_boxes)

            obj_order_ids = torch.arange(n_boxes, dtype=torch.long, device=device)
            obj_order_ids = torch.tensor(self.tokenizer.encode([f'<vis_extra_id_{index}>' for index in obj_order_ids], add_special_tokens=False))
            obj_order_ids = obj_order_ids.view(1, 1, n_boxes).expand(B, context_images_amount, -1).contiguous().view(B, context_images_amount*n_boxes).to(device)
            
            output = self(input_ids=input_ids,
                        vis_inputs=(vis_feats, vis_pos, img_order_ids, obj_order_ids),
                        vis_attention_mask=vis_attention_mask,
                        labels=labels,
                        return_dict=True)
        else:
            output = self(input_ids=input_ids,
                        labels=labels,
                        return_dict=True)

        # Don't take padding tokens into account for loss
        lm_mask = (labels != -100).float()
        B, L = labels.size()

        loss = output['loss']
        loss = loss.view(B, L) * lm_mask
        loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)  # B
        loss = loss.mean()

        result = {
            'loss': loss
        }

        return result

    @torch.no_grad()
    def valid_step(self, batch):
        self.eval()

        device = next(self.parameters()).device
        input_ids = batch['input_ids'].to(device)
        B = len(input_ids)
        
        labels = batch['target_ids'].to(device)
       
        if not self.config.just_text_features:
            feat_dim = batch['vis_feats'].size(3)
            n_boxes = batch['vis_feats'].size(2)
            context_images_amount = batch['vis_feats'].size(1)
            
            vis_feats = batch['vis_feats'].to(device).view(B, context_images_amount*n_boxes, feat_dim)
            vis_pos = batch['boxes'].to(device).view(B, context_images_amount*n_boxes, 4)
            vis_attention_mask = batch['vis_attention_mask'].to(device).view(B, context_images_amount*n_boxes)

            img_order_ids = batch['img_order_ids'].to(device).view(B, context_images_amount*n_boxes)


            if self.config.match_text_image:
                img_order_ids = batch['img_order_encoded'].to(device).view(B, context_images_amount*n_boxes)
            obj_order_ids = torch.arange(n_boxes, dtype=torch.long, device=device)
            obj_order_ids = torch.tensor(self.tokenizer.encode([f'<vis_extra_id_{index}>' for index in obj_order_ids], add_special_tokens=False))
            obj_order_ids = obj_order_ids.view(1, 1, n_boxes).expand(B, context_images_amount, -1).contiguous().view(B, context_images_amount*n_boxes).to(device)
            
            output = self(input_ids=input_ids,
                        vis_inputs=(vis_feats, vis_pos, img_order_ids, obj_order_ids),
                        vis_attention_mask=vis_attention_mask,
                        labels=labels,
                        return_dict=True)
        else:
            output = self(input_ids=input_ids,
                        labels=labels,
                        return_dict=True)

        # Don't take padding tokens into account for loss
        lm_mask = (labels != -100).float()
        B, L = labels.size()

        loss = output['loss']
        loss = loss.view(B, L) * lm_mask
        loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)  # B

        result = {
            'loss': loss.cpu().tolist()
        }

        return result

    @torch.no_grad()
    def test_step(self, batch, **kwargs):
        self.eval()

        device = next(self.parameters()).device
        input_ids = batch['input_ids'].to(device)
        B = len(input_ids)
        decoder_input_ids = torch.ones(B, 1, dtype=torch.long, device=device) * self.config.decoder_start_token_id
               
        if not self.config.just_text_features:
            feat_dim = batch['vis_feats'].size(3)
            n_boxes = batch['vis_feats'].size(2)
            context_images_amount = batch['vis_feats'].size(1)
            
            vis_feats = batch['vis_feats'].to(device).view(B, context_images_amount*n_boxes, feat_dim)
            vis_pos = batch['boxes'].to(device).view(B, context_images_amount*n_boxes, 4)
            vis_attention_mask = batch['vis_attention_mask'].to(device).view(B, context_images_amount*n_boxes)

            img_order_ids = batch['img_order_ids'].to(device).view(B, context_images_amount*n_boxes)
            if self.config.match_text_image:
                img_order_ids = batch['img_order_encoded'].to(device).view(B, context_images_amount*n_boxes)
         
            obj_order_ids = torch.arange(n_boxes, dtype=torch.long, device=device)
            obj_order_ids = torch.tensor(self.tokenizer.encode([f'<vis_extra_id_{index}>' for index in obj_order_ids], add_special_tokens=False))
            obj_order_ids = obj_order_ids.view(1, 1, n_boxes).expand(B, context_images_amount, -1).contiguous().view(B, context_images_amount*n_boxes).to(device)   

            output = self.generate(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos, img_order_ids, obj_order_ids),
                vis_attention_mask=vis_attention_mask,
                decoder_input_ids=decoder_input_ids,
                **kwargs
            )
        else:
            output = self.generate(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                **kwargs
            )
        output = [element[element.index(self.tokenizer.bos_token_id)+1:element[1:].index(self.tokenizer.eos_token_id)+1]for element in output.cpu().tolist()]
        generated_sents = self.tokenizer.batch_decode(output)
        
        return {'token_ids': output,
                'pred': generated_sents}

