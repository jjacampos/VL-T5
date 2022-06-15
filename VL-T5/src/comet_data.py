from collections import defaultdict
import re
import os
import random
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, Sampler
import pdb
from transformers import T5TokenizerFast, BartTokenizer
from tokenization import VLT5TokenizerFast

MEMORY_BREAK = "<MM_BREAK>"
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

class COMETFineTuneDataset(Dataset):

    def __init__(self, raw_dataset, coco_mapping, coco_features, args, tokenizer, verbose=True, randomized_indexes=False, num_turns=2):
        super().__init__()

        self.raw_dataset = raw_dataset
        self.coco_mapping = coco_mapping
        self.coco_features = coco_features
        self.max_images = args.n_images
        
        # Features hyperparams
        self.n_boxes = args.n_boxes
        self.feat_dim = args.feat_dim
        
        self.tokenizer = tokenizer

        self.randomized_indexes = randomized_indexes
        self.num_turns = num_turns

        self.args = args
        
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
        

    # TODO many things can be moved to initialization
    def __getitem__(self, idx):

        example = self.raw_dataset[idx]
        # Get non repeated but ordered memory ids from input
        memory_ids = []
        
        # We want memories in inverse order to ensure that last appeared are in features. 
        for element in re.findall(f'(\d+)', example['predict'])[::-1]:
            if int(element) in self.coco_mapping and int(element) not in memory_ids:
                memory_ids.append(int(element))

        order = [i for i in range(len(memory_ids))]
        if self.randomized_indexes:
            random.shuffle(order)
            
        # Get the memory features
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

        img_order_ids = torch.LongTensor(img_order_ids).view(max(1, min(self.max_images, len(memory_ids))), self.n_boxes)

        # Get the text features and remove the MEMORY BREAK tag
        input_sentence = example['predict']
        target_sentence = example['target'].replace(MEMORY_BREAK, '')
        
        # Cut the amount of turns
        input_sentence = USER.join(input_sentence.split(USER)[-self.num_turns+1:])
        # Add USER tag at begining if we removed it
        input_sentence = USER + input_sentence if input_sentence[:6] != USER else input_sentence

        # Use local context for image ids
        for index, memory in enumerate(memory_ids):
            input_sentence = input_sentence.replace(f'{memory}', f'mem_id_{order[index]}')
            target_sentence = target_sentence.replace(f'{memory}', f'mem_id_{order[index]}')
            
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
        return out_dict


    def collate_fn(self, batch):

        # Do padding for text and images
        B = len(batch)
        num_boxes = batch[0]['boxes'].size(1)
        max_input_len = max(len(entry['input_ids']) for entry in batch)
        max_target_len = max(len(entry['target_ids']) for entry in batch)
        max_images_context = max(entry['vis_feats'].size(0) for entry in batch)
        feats_dim = batch[0]['vis_feats'].size(-1)

        input_ids = torch.ones(B, max_input_len, dtype=torch.long) * self.tokenizer.pad_token_id

        boxes = torch.zeros(B, max_images_context, num_boxes, 4, dtype=torch.float)
        vis_feats = torch.zeros(B, max_images_context, num_boxes, feats_dim, dtype=torch.float)
        vis_attention_mask = torch.zeros(B, max_images_context, num_boxes, dtype=torch.float)
        target_ids = torch.ones(B, max_target_len, dtype=torch.long) * self.tokenizer.pad_token_id
        # Use pad token for img_order ids padding. 
        img_order_ids = torch.ones(B, max_images_context, num_boxes, dtype=torch.long) * self.tokenizer.pad_token_id

        for i, entry in enumerate(batch):

            # If the amount of context images is greater than one
            if not(entry['boxes'].size(0) == 1 and torch.all(entry['boxes']==0)):
                vis_attention_mask[i,:entry['boxes'].size(0)] = 1
            input_ids[i, :len(entry['input_ids'])] = entry['input_ids']
            boxes[i,:entry['boxes'].size(0)] += entry['boxes']
            vis_feats[i,:entry['vis_feats'].size(0)] += entry['vis_feats']
            target_ids[i, :len(entry['target_ids'])] = entry['target_ids']
            img_order_ids[i, :entry['img_order_ids'].size(0)] = entry['img_order_ids']

        word_mask = target_ids != self.tokenizer.pad_token_id
        target_ids[~word_mask] = -100

        return {'boxes': boxes,
                'vis_feats': vis_feats,
                'vis_attention_mask': vis_attention_mask,
                'input_ids': input_ids,
                'target_ids': target_ids,
                'examples': [elem['example'] for elem in batch],
                'img_order_ids': img_order_ids}
    

class COMETEvaluator:
    
    def evaluate(self, predicts, examples, test_file, output_path):

        test_file = test_file.replace('_gpt2', '')
        
        # Recover global indexes
        for i in range(len(predicts)):
            cur_pred = predicts[i].replace(END_OF_API_CALL, '').replace(END_OF_SENTENCE, '').replace('< EOAC>', '').replace('< EOS>', '')
            for mapping in examples[i]['mapping']:
                cur_pred = cur_pred.replace(f'mem_id_{mapping[1]}', f'{mapping[0]}')
            # If there are still mem_ids in the pred but there were no mapping in context remove them. 
            remaining = re.findall('(mem_id_[0-9]*)', predicts[i])
            for to_remove in remaining:
                cur_pred = cur_pred.replace(to_remove, '')
            examples[i]['model_prediction'] = cur_pred

        # Output the model predictions
        json.dump(examples, open(os.path.join(output_path, 'predictions.json'), 'w'))
        
        # Call the evaluation scripts
        os.system(f"python3 /data/home/jacampos/project/updated_vl/VL-T5/VL-T5/evaluation/create_results_json_memory.py \
            --memory_test_json {test_file} --model_output_json {os.path.join(output_path, 'predictions.json')}")

        os.system(f"python3 /data/home/jacampos/project/updated_vl/VL-T5/VL-T5/evaluation/response_evaluation_memory.py \
        --data_json_path {test_file} --model_response_path {os.path.join(output_path, 'predictions_response_results.json')}")

        os.system(f"python3 /data/home/jacampos/project/updated_vl/VL-T5/VL-T5/evaluation/evaluate_dst_memory.py \
        --input_path_target {test_file} --input_path_predicted {os.path.join(output_path, 'predictions_dst_results.json')}\
        --output_path_report {os.path.join(output_path, 'report.out')}")

        output_report = json.load(open(os.path.join(output_path, 'report.out'), 'r'))

        return output_report
        


        
        


        
            
        
        
        
        
