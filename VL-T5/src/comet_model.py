import torch
import torch.nn as nn
import numpy as np


from modeling_t5 import VLT5

class VLT5COMET(VLT5):
    def __init__(self, config):
        super().__init__(config)

    def train_step(self, batch):

        device = next(self.parameters()).device

        input_ids = batch['input_ids'].to(device)
        B = len(input_ids)

        vis_feats = batch['vis_feats'].to(device)
        vis_pos = batch['boxes'].to(device)
        context_images_amount = vis_feats.size(1)
        feat_dim = vis_feats.size(2)
        
        labels = batch['target'].to(device)

        img_order_ids = []
        for i in range(context_images_amount):
            image_order_ids += [i] * feat_dim
            
        img_order_ids = torch.tensor(img_order_ids, dtype=torch.long, device=device)
        img_order_ids = img_order_ids.view(1, context_images_amount*feat_dim).expand(B, -1)

        obj_order_ids = torch.arange(V_L, dtype=torch.long, device=device)
        obj_order_ids = obj_order_ids.view(1, 1, V_L).expand(B, context_images_amount, -1).contiguous().view(B, 2*V_L)

        
        output = self(input_ids=input_ids,
                      vis_inputs=(vis_feats, vis_pos, img_order_ids, obj_order_ids),
                      labels=labels,
                      return_dict=True)


        # Don't take padding tokens into account for loss
        lm_mask = (lm_labels != -100).float()
        B, L = lm_labels.size()

        loss = output['loss']
        loss = loss.view(B, L) * lm_mask
        loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)  # B
        loss = loss * batch['scores'].to(device=device)
        loss = loss.mean()

        result = {
            'loss': loss
        }

        return result

    @torch.no_grad()
    def test_step(self, batch, **kwargs):
        self.eval()
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)
        
        output = self.generate(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            **kwargs
        )
        generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        result['token_ids'] = output
        result['pred_ans'] = generated_sents

        return result
        
        
from modeling_bart import VLBart

class VLBartComet(VLBart):

    def __init__(self, config):
        super().__init__(config)


    def train_step(self, batch):

        device = next(self.parameters()).device

        input_ids = batch['input_ids'].to(device)
        B = len(input_ids)

        vis_feats = batch['vis_feats'].to(device)
        vis_pos = batch['boxes'].to(device)
        context_images_amount = vis_feats.size(1)
        feat_dim = vis_feats.size(2)
        
        labels = batch['target'].to(device)

        img_order_ids = []
        for i in range(context_images_amount):
            image_order_ids += [i] * feat_dim
            
        img_order_ids = torch.tensor(img_order_ids, dtype=torch.long, device=device)
        img_order_ids = img_order_ids.view(1, context_images_amount*feat_dim).expand(B, -1)

        obj_order_ids = torch.arange(V_L, dtype=torch.long, device=device)
        obj_order_ids = obj_order_ids.view(1, 1, V_L).expand(B, context_images_amount, -1).contiguous().view(B, 2*V_L)

        
        output = self(input_ids=input_ids,
                      vis_inputs=(vis_feats, vis_pos, img_order_ids, obj_order_ids),
                      labels=labels,
                      return_dict=True)


        # Don't take padding tokens into account for loss
        lm_mask = (lm_labels != -100).float()
        B, L = lm_labels.size()

        loss = output['loss']
        loss = loss.view(B, L) * lm_mask
        loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)  # B
        loss = loss * batch['scores'].to(device=device)
        loss = loss.mean()

        result = {
            'loss': loss
        }

        return result

    @torch.no_grad()
    def test_step(self, batch, **kwargs):
        self.eval()
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)
        
        output = self.generate(
            input_ids=input_ids,
            vis_inputs=(vis_feats, vis_pos),
            **kwargs
        )
        generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        result['token_ids'] = output
        result['pred_ans'] = generated_sents

        return result

        
