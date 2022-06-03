import json
import torch
import numpy as np

class COMETFineTuneDataset(Dataset):

    def __init__(self, raw_dataset, coco_mapping, coco_features, args, verbose=True):
        super().__init()
        
        self.raw_dataset = raw_dataset
        self.coco_mapping = coco_mapping
        self.coco_features = coco_features

        # Features hyperparams
        self.n_boxes = args.n_boxes
        self.feat_dim = args.feat_dim
        
        if self.verbose:
            print('Data sources: ', self.sources)

        if 't5' in self.args.backbone:
            self.tokenizer = VLT5TokenizerFast.from_pretrained(
                args.backbone,
                max_length=self.args.max_text_length,
                do_lower_case=self.args.do_lower_case)
        elif 'bart' in self.args.backbone:
            self.tokenizer = BartTokenizer.from_pretrained(
                args.backbone,
                do_lower_case=self.args.do_lower_case)

    def __len__(self):
        return len(self.data)

    def _get_image_features(self, memory_ids):
        
        # Get image features
        feats_list = []
        boxes_list = []
        for memory_id in memory_ids:
            # Features
            img_id = self.coco_mapping[memory_id]
            feats = np.zeros(shape=(self.n_boxes, self.feat_dim), dtype=np.float32)
            self.coco_features[f'{img_id}/features'].read_direct(feats)
            feats_list.append(feats)

            # BBoxes
            img_h = self.coco_features[f'{img_id}/img_h'][()]
            img_w = self.coco_features[f'{img_id}/img_w'][()]
            img_w = self.coco_features[f'{img_id}/img_w'][()]
            boxes = self.coco_features[f'{img_id}/boxes'][()]  # (x1, y1, x2, y2)
            boxes[:, (0, 2)] /= img_w
            boxes[:, (1, 3)] /= img_h
            np.testing.assert_array_less(boxes, 1+1e-5)
            np.testing.assert_array_less(-boxes, 0+1e-5)
            boxes_list.append(boxes)
            
        feats = np.stack(feats_list) # [len(memory_ids), n_boxes, feat_dim]
        feats = torch.from_numpy(feats)
        boxes = np.stack(boxes_list)
        boxes = torch.from_numpy(boxes)
        boxes.clamp_(min=0.0, max=1.0)

        return feats, boxes
        
    
    def __getitem__(self, idx):

        example = self.raw_dataset[idx]
        
        # Get memory ids from input
        split_str = instance['predict'].split(MEMORY_BREAK)
        memory_ids = [int(element.rsplit(" ", 1)[-1]) for element in split_str[:-1]]

        # Get the memory features
        feats, boxes = self._get_image_features(memory_ids)
        out_dict = {'boxes':boxes,
                    'vis_feats': feats}
        
        # Get the text features
        input_sentence = example['predict']
        target_sentence = example['target']
        # TODO use different tokens for API and normal generation now just using "comet" as input
        input_ids = self.tokenizer.encode(f'comet: {input_sentence}')
        out_dict['input_ids'] = torch.LongTensor(input_ids)

        target_ids = self.tokenizer.encode(target_sentence)
        out_dict['target_ids'] = torch.LongTensor(target_ids)
        
        return out_dict


    def collate_fn(self, batch):

        # Do padding for text and images
        B = len(batch)
        num_boxes = batch[0]['boxes'].size(1)
        max_input_len = max(len(entry['input_ids']) for entry in batch)
        max_target_len = max(len(entry['target_ids']) for entry in batch)
        max_images_context = max(entry['vis_feats'].size(1) for entry in batch)
        feats_dim = batch[0]['vis_feats'].size(-1)

        input_ids = torch.ones(B, max_input_len, dtype=torch.long) * self.tokenizer.pad_token_id

        boxes = torch.zeors(B, max_images_context, num_boxes, 4, dtype=torch.float)
        vis_feats = torch.zeros(B, max_images_context, V_L, feat_dim, dtype=torch.float)

        targets = torch.zeros(B, max_target_len, dtype=torch.long) * self.tokenizer.pad_token_id

        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']

            boxes[i] += entry['boxes']
            vis_feats[i] += entry['vis_feats']
            target_ids[i, :entry['target_length']] = entry['target_ids']

        
        
        return {'boxes': boxes,
                'vis_feats': vis_feats,
                'input_ids': input_ids,
                'target': target_ids}
    

        
        


        
            
        
        
        
        
