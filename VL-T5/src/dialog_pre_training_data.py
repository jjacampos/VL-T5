from collections import defaultdict
import copy
import re
import os
import random
import json
import torch
from tqdm import tqdm
import numpy as np
import stanza
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import T5TokenizerFast, BartTokenizer
from tokenization import VLT5TokenizerFast

START_OF_MULTIMODAL_CONTEXTS = '<SOM>'
END_OF_MULTIMODAL_CONTEXTS = '<EOM>'
START_OF_API_CALL = '=> <SOAC>:'
END_OF_API_CALL = '<EOAC>'
START_OF_API_RESULT = '<SOAR>'
END_OF_API_RESULT = '<EOAR>'
START_OF_RESPONSE = "<SOR>"
END_OF_SENTENCE = '<EOS>'
SYSTEM = '<SYSTEM>'
USER = '<USER>'

class DialPreFineTuneDataset(Dataset):

    def __init__(self, raw_dataset, coco_mapping, coco_features, args, tokenizer, randomization_type, verbose=True):
        super().__init__()

        self.raw_dataset = raw_dataset
        self.coco_mapping = coco_mapping
        self.coco_features = coco_features
        self.max_images = args.n_images
        
        # Features hyperparams
        self.n_boxes = args.n_boxes
        self.feat_dim = args.feat_dim
        
        self.tokenizer = tokenizer
        self.losses = args.dialog_losses.split(',')
        print(self.losses)

        self.randomization_type = randomization_type
        self.num_turns = args.num_turns

        self.args = args

        new_dataset = []

        all_user_utterances, all_system_utterances, all_user_entities, all_system_entities = [], [], [], []

        for datum in self.raw_dataset:
            input_sentence = datum['predict']
            user_sentences = [f"<USER> {elem}".strip() for elem in re.findall(f'<USER>(.*?)<SYSTEM>', input_sentence)]
            system_sentences = [f"<SYSTEM> {elem[0]}".strip() for elem in re.findall(f'<SYSTEM>(.*?)(<USER>|<EOS>)', input_sentence)]
            all_user_utterances += user_sentences
            all_system_utterances += system_sentences
            all_user_entities.append(datum["entities"][0::2])
            all_system_entities.append(datum["entities"][1::2])

        for datum in self.raw_dataset:
            current_loss = random.sample(self.losses, 1)[0]
            input_sentence = datum['predict']
            current_user_sentences = [f"<USER> {elem}".strip() for elem in re.findall(f'<USER>(.*?)<SYSTEM>', input_sentence)]
            current_system_sentences = [f"<SYSTEM> {elem[0]}".strip() for elem in re.findall(f'<SYSTEM>(.*?)(<USER>|<EOS>)', input_sentence)]
            current_sentences = []
            current_memories = [memory.strip() for memories in re.findall('<SOM> (.*?) <EOM>', input_sentence) for memory in memories.split(',') if len(memory) > 0 ]
            for user_ut, system_ut in zip(current_user_sentences, current_system_sentences):
                current_sentences.append(user_ut)
                current_sentences.append(system_ut)
            dialog = " ".join(current_sentences)
            dialog_len = len(current_sentences)

            if "speaker" == current_loss:
                for user_utterance in current_user_sentences:
                    new_dataset.append({
                        'predict': f"speaker: {user_utterance.replace('<USER>', '').strip()}",
                        'target': '<USER>',
                        'task': 'speaker'})
                for system_utterance in current_system_sentences:
                    # We don't want memories to appear in this task because it would be trivial for the system
                    new_dataset.append({
                        'predict': f"speaker: {system_utterance.replace('<SYSTEM>', '').split('<SOM>')[0].strip()}",
                        'target': '<SYSTEM>',
                        'task': 'speaker'}) 

            if "coherence" == current_loss:
                input_sentence = datum['predict']
                if random.uniform(0,1) > 0.5:
                    utterances_to_modify = random.sample(range(0, dialog_len), random.randint(1, dialog_len))
                    input_example = ""
                    for i in range(dialog_len):
                            if i in utterances_to_modify:
                                if "<USER>" in current_sentences[i]:
                                    input_example += f"{random.sample(all_user_utterances, 1)[0]} "
                                else:
                                    input_example += f"{random.sample(all_system_utterances, 1)[0]} "
                            else:
                                input_example += f"{current_sentences[i]} "
                    target = 'false'
                else:
                    input_example = dialog                             
                    target = 'true'               
                new_dataset.append({ 
                    'predict': f"coherence: {input_example.strip()}", 
                    'target': target,
                    'task': 'coherence'
                    })      

            if "reordering" == current_loss:
                shuffled_sentences = random.sample(current_sentences, len(current_sentences))
                original_input = dialog
                shuffled_target = " ".join(shuffled_sentences)
                new_dataset.append({
                    'predict': f"reordering: {shuffled_target}",
                    'target': original_input,
                    'task': 'reordering'})

            if "entities" == current_loss:
                for sentence, entities in zip(current_sentences, datum['entities']):
                    if len(entities) == 0:
                        continue
                    new_dataset.append({ 
                        'predict': f"entities: {sentence}",
                        'target': ",".join(entities) if args.use_entity_types else ','.join([elem.split(':')[0] for elem in entities]),
                        'task': 'entities'}
                        )

            if "mm_coherence" == current_loss and len(current_memories) > 0:
                if random.uniform(0,1) > 0.5:
                    memories_to_modify = random.sample(range(0, len(current_memories)), random.randint(1, len(current_memories)))
                    input_example = copy.deepcopy(dialog)
                    for index in memories_to_modify:
                        input_example = input_example.replace(current_memories[index], str(random.sample(self.coco_mapping.keys(), 1)[0]))
                    target = "false"
                else:
                    input_example = dialog
                    target = "true"
                new_dataset.append({ 
                    'predict': f"coherence: {input_example}", 
                    'target': target,
                    'task': 'mm_coherence'
                    })  

            if "mm_reordering" == current_loss and len(current_memories) > 0:
                shuffled_memories = random.sample(current_memories, len(current_memories))
                input_example = copy.deepcopy(dialog)
                for orig_memory, shuffled_memory in zip(current_memories, shuffled_memories):
                    # Trick that avoid re substituting a string
                    input_example = re.sub(f'{orig_memory}([^_])', f'{shuffled_memory}_new\g<1>', input_example)
                input_example = input_example.replace('_new', '')
                new_dataset.append({
                    'predict': f"reordering: {input_example}",
                    'target': dialog,
                    'task': 'mm_reordering'})
                
        self.raw_dataset = new_dataset

    def __len__(self):
        return len(self.raw_dataset)

    def _get_image_features(self, memory_ids):
        
        # Get image features
        feats_list = []
        boxes_list = []

        for memory_id in memory_ids[:self.max_images]:
            # Features
            img_id = self.coco_mapping[memory_id].split('.jpg')[0]
            feats = np.zeros(shape=(self.n_boxes, self.feat_dim), dtype=np.float32)
            # Get the top n_boxes taking prob into account
            indexes = (-self.coco_features[f'{img_id}/attr_conf'][()]).argsort()[:self.n_boxes]
            feats = self.coco_features[f'{img_id}/features'][()][indexes]
            #self.coco_features[f'{img_id}/features'][()][indexes].read_direct(feats)
            feats_list.append(feats)

            # BBoxes
            img_h = self.coco_features[f'{img_id}/img_h'][()]
            img_w = self.coco_features[f'{img_id}/img_w'][()]
            boxes = self.coco_features[f'{img_id}/boxes'][()][indexes]  # (x1, y1, x2, y2)
            boxes[:, (0, 2)] /= img_w
            boxes[:, (1, 3)] /= img_h
            np.testing.assert_array_less(boxes, 1+1e-5)
            np.testing.assert_array_less(-boxes, 0+1e-5)
            boxes_list.append(boxes)

        if len(feats_list) == 0:
            # Add one zeros image for the case in which we don't have context memories
            feats = np.zeros(shape=(self.n_boxes, self.feat_dim), dtype=np.float32)
            feats_list.append(feats)
            # BBoxes
            boxes = np.zeros(shape=(self.n_boxes, 4))
            boxes_list.append(boxes)
            
        feats = np.stack(feats_list) # [len(memory_ids), n_boxes, feat_dim]
        feats = torch.from_numpy(feats)
        boxes = np.stack(boxes_list)
        boxes = torch.from_numpy(boxes)
        boxes.clamp_(min=0.0, max=1.0)

        return feats, boxes
        
    def _generate_obj_ids_(self, img_ids, nboxes):
        obj_ids = []
        for img_id in img_ids:
            for i in range(nboxes):
                obj_ids.append(img_id * nboxes + i)
        return obj_ids

    # TODO many things can be moved to initialization
    def __getitem__(self, idx):

        example = self.raw_dataset[idx]

        # Get the text features 
        input_sentence = example['predict']
        target_sentence = example['target']
        
        # Cut the amount of turns
        input_sentence = USER.join(input_sentence.split(USER)[-self.num_turns-1:])
        # Add USER tag at begining if we removed it
        input_sentence = USER + input_sentence if input_sentence[:6] != USER else input_sentence

        # Get non repeated but ordered memory ids from input
        memory_ids = []

        # We want memories in inverse order to ensure that last ones are added as features. 
        for element in re.findall(f'(\d+)', input_sentence)[::-1]:
            if int(element) in self.coco_mapping and int(element) not in memory_ids:
                memory_ids.append(int(element))

        order = [i for i in range(self.max_images)]

        if self.randomization_type == 'random_global':
            random.shuffle(order)
        elif self.randomization_type == 'random_local':
            for_randomization = order[:len(memory_ids)]
            random.shuffle(for_randomization)
            order[:len(memory_ids)] = for_randomization

        out_dict = {}
        # Get the memory features
        if not self.args.just_text_features:
            feats, boxes = self._get_image_features(memory_ids)
            out_dict = {'boxes':boxes,
                    'vis_feats': feats}
                    
        # Get the img_order_ids
        img_order_ids = []
        for i in range(min(self.max_images, len(memory_ids))):
            img_order_ids += [order[i]] * self.n_boxes

        # Add one padding if we don't have memories in context
        if len(img_order_ids) == 0:
            img_order_ids += [self.tokenizer.pad_token_id] * self.n_boxes
        img_order_ids_old = img_order_ids
        img_order_encoded = self.tokenizer.encode([f'<img_extra_id_{index}>' for index in img_order_ids], add_special_tokens=False)
        img_order_ids = torch.LongTensor(img_order_ids).view(max(1, min(self.max_images, len(memory_ids))), self.n_boxes)
        img_order_encoded = torch.LongTensor(img_order_encoded).view(max(1, min(self.max_images, len(memory_ids))), self.n_boxes)

        # Use local context for image ids
        for index, memory in enumerate(memory_ids[:self.max_images]):
            input_sentence = input_sentence.replace(f'{memory}', f' <img_extra_id_{order[index]}> ')
            target_sentence = target_sentence.replace(f'{memory}', f' <img_extra_id_{order[index]}> ')

        # If the target memory is not in the context just use mem_id_99 as unknown for consistency
        for memory in re.findall(f'(\d+)', target_sentence):
            if int(memory) in self.coco_mapping:
                target_sentence = target_sentence.replace(f'{memory}', ' <img_extra_id_99> ')
        
        # TODO use different tokens for API and normal generation now just using "comet" as input
        input_ids = self.tokenizer.encode(f'comet: {input_sentence}', \
            max_length=self.args.max_text_length, truncation=True)
        out_dict['input_ids'] = torch.LongTensor(input_ids)

        target_ids = self.tokenizer.encode(target_sentence)
        out_dict['target_ids'] = torch.LongTensor(target_ids)
        # Get the mapping back from id to memory        
        example['mapping'] = [(memory, idx) for memory, idx in zip(memory_ids, order)]
        out_dict['example'] = example
        out_dict['img_order_ids'] = img_order_ids
        out_dict['img_order_encoded'] = img_order_encoded
        out_dict['obj_order_ids'] = torch.LongTensor(self.tokenizer.encode([f'<vis_extra_id_{index}>' for index in self._generate_obj_ids_(list(set(img_order_ids_old)), self.n_boxes)]\
            , add_special_tokens=False))
        out_dict["task"] = example["task"]
        return out_dict


    def collate_fn(self, batch):

        # Do padding for text and images
        B = len(batch)
        max_input_len = max(len(entry['input_ids']) for entry in batch)
        max_target_len = max(len(entry['target_ids']) for entry in batch)
        input_ids = torch.ones(B, max_input_len, dtype=torch.long) * self.tokenizer.pad_token_id  
        target_ids = torch.ones(B, max_target_len, dtype=torch.long) * self.tokenizer.pad_token_id      

        if not self.args.just_text_features:
            num_boxes = batch[0]['boxes'].size(1)
            max_images_context = max(entry['vis_feats'].size(0) for entry in batch)
            feats_dim = batch[0]['vis_feats'].size(-1)
            boxes = torch.zeros(B, max_images_context, num_boxes, 4, dtype=torch.float)
            vis_feats = torch.zeros(B, max_images_context, num_boxes, feats_dim, dtype=torch.float)
            vis_attention_mask = torch.zeros(B, max_images_context, num_boxes, dtype=torch.float)
            # Use pad token for img_order ids padding. 
            img_order_ids = torch.ones(B, max_images_context, num_boxes, dtype=torch.long) * self.tokenizer.pad_token_id
            img_order_encoded = torch.ones(B, max_images_context, num_boxes, dtype=torch.long) * self.tokenizer.pad_token_id
            obj_order_ids = torch.ones(B, max_images_context * num_boxes, dtype=torch.long) * self.tokenizer.pad_token_id

        for i, entry in enumerate(batch):
            input_ids[i, :len(entry['input_ids'])] = entry['input_ids']            
            target_ids[i, :len(entry['target_ids'])] = entry['target_ids']                

            if not self.args.just_text_features:
                # If the amount of context images is greater than one
                if not(entry['boxes'].size(0) == 1 and torch.all(entry['boxes']==0)):
                    vis_attention_mask[i,:entry['boxes'].size(0)] = 1
                boxes[i,:entry['boxes'].size(0)] += entry['boxes']
                vis_feats[i,:entry['vis_feats'].size(0)] += entry['vis_feats']
                img_order_ids[i, :entry['img_order_ids'].size(0)] = entry['img_order_ids']
                img_order_encoded[i, :entry['img_order_ids'].size(0)] = entry['img_order_encoded']
                obj_order_ids[i, :entry['obj_order_ids'].size(0)] = entry['obj_order_ids']

        word_mask = target_ids != self.tokenizer.pad_token_id
        target_ids[~word_mask] = -100

        out_dict = {'input_ids': input_ids,
                'target_ids': target_ids,
                'examples': [elem['example'] for elem in batch],
                "task": [elem['task'] for elem in batch]
                }
    
        if not self.args.just_text_features:
            out_dict =  {'boxes': boxes,
                'vis_feats': vis_feats,
                'vis_attention_mask': vis_attention_mask,
                'img_order_ids': img_order_ids,
                'input_ids': input_ids,
                'target_ids': target_ids,
                'img_order_encoded': img_order_encoded,
                'examples': [elem['example'] for elem in batch],
                "task": [elem['task'] for elem in batch],
                'obj_order_ids':obj_order_ids}
    
        return out_dict

