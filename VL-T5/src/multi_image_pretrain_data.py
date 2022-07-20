from asyncio.constants import LOG_THRESHOLD_FOR_CONNLOST_WRITES
from email.policy import default
from cv2 import filterHomographyDecompByVisibleRefpoints
from torch.utils.data import DataLoader, Dataset, Sampler
from pathlib import Path
from collections import defaultdict
import json
import random
from multiprocessing import Pool
import h5py
import pickle
import math
from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data.distributed import DistributedSampler
from copy import deepcopy


from transformers import T5Tokenizer, BartTokenizer, T5TokenizerFast, BartTokenizerFast
from tokenization import VLT5Tokenizer, VLT5TokenizerFast

import preprocess
from qa_answer_table import AnswerTable

project_dir = Path(__file__).resolve().parent.parent # VLT5
workspace_dir = project_dir.parent
dataset_dir = Path("/fsx/jacampos/data/pretraining/datasets")
coco_dir = dataset_dir.joinpath('COCO')
vg_dir = dataset_dir.joinpath('VG')
coco_img_dir = coco_dir.joinpath('images/')


# Load VG Classes
vg_classes = []
with open(vg_dir.joinpath('objects_vocab.txt')) as f:
    for obj in f.readlines():
        vg_classes.append(obj.split(',')[0].lower().strip())

vg_attrs = []
with open(vg_dir.joinpath('attributes_vocab.txt')) as f:
    for attr in f.readlines():
        vg_attrs.append(attr.split(',')[0].lower().strip())

def make_uid(img_id, dset, sent_idx):
    return "%s_%s_%03d" % (img_id, dset, sent_idx)


def get_datum(datum, max_num_captions=5):

    data = []
    _sents = []

    args = datum['args']
    # Control that we just add the elem once for image captioning and image grounding
    added = False
    if datum['is_train']:
        if 'COCO_train2014' in datum['img_id']:
            img_source = 'mscoco_resplit_train_train2014'
        elif 'COCO_val2014' in datum['img_id']:
            img_source = 'mscoco_resplit_train_val2014'
        else:
            img_source = 'vgnococo'
    else:
        img_source = 'mscoco_resplit_val'

    for text_source, sents in datum['sentf'].items():
        if datum['caption_only']:
            if text_source not in ['mscoco', 'vg']:
                continue

        if args.coco_only:
            if text_source != 'mscoco':
                continue

        img_id = datum['img_id']
        labels = None
        if datum['qa'] and text_source in datum['labelf']:
            labels = datum['labelf'][text_source]
        else:
            if not added:
                if datum['ci']:
                    for i in range(args.ground_upsample):
                        new_datum = {
                            'uid': make_uid(img_id, 'ci', i),
                            'img_id': img_id,
                            'img_source': img_source,
                            'task': 'ci',
                            'text_source': text_source,
                            'sent': random.choice(sents),
                            'label': None,
                        }
                        data.append(new_datum)
                if datum['ig']:
                    for i in range(args.ground_upsample):
                        new_datum = {
                            'uid': make_uid(img_id, 'ig', i),
                            'img_id': img_id,
                            'img_source': img_source,
                            'task': 'ig',
                            'text_source': text_source,
                            'sent': random.choice(sents),
                            'label': None,
                        }
                        data.append(new_datum)
                    added = True


        for sent_idx, sent in enumerate(sents[:max_num_captions]):

            if ('t5' in datum['backbone'] or 'bart' in datum['backbone']) and len(sent.split()) <= 2:
                continue

            # remove duplicate sentence
            if sent in _sents:
                continue

            new_datum = {
                'uid': make_uid(img_id, text_source, sent_idx),
                'img_id': img_id,
                'img_source': img_source,
                'sent': sent,
                'text_source': text_source
            }

            # Task: QA
            if datum['qa'] and labels is not None:
                label = labels[sent_idx]
                if ('t5' in datum['backbone'] or 'bart' in datum['backbone']) and len(label) == 0:
                    continue
                else:
                    # assert len(label) > 0, (img_id, labels, sent_idx, label)
                    # can have len = 0
                    new_datum = deepcopy(new_datum)
                    new_datum['task'] = 'qa'
                    new_datum['label'] = label
                    data.append(new_datum)

            # Task: Language modeling
            if datum['lm'] and labels is None:
                new_datum = deepcopy(new_datum)
                new_datum['task'] = 'lm'
                new_datum['label'] = None
                data.append(new_datum)

            # Task: Image captioning
            if datum['caption']:
                if args.caption_cocoonly:
                    if text_source == 'mscoco':
                        new_datum = deepcopy(new_datum)
                        new_datum['task'] = 'caption'
                        new_datum['label'] = None
                        data.append(new_datum)
                else:
                    if text_source in ['mscoco', 'vg']:
                        new_datum = deepcopy(new_datum)
                        new_datum['task'] = 'caption'
                        new_datum['label'] = None
                        data.append(new_datum)

            _sents.append(sent)

    if datum['cr']:
        for i in range(args.ground_upsample):
            new_datum = {
                'uid': make_uid(img_id, 'cr', i),
                'img_id': img_id,
                'img_source': img_source,
                'task': 'cr',
                'text_source': 'cr',
                'sent': None,
                'label': None,
            }
            data.append(new_datum)


    if datum['og']:
        for i in range(args.ground_upsample):
            new_datum = {
                'uid': make_uid(img_id, 'og', i),
                'img_id': img_id,
                'img_source': img_source,
                'task': 'og',
                'text_source': 'og',
                'sent': None,
                'label': None,
            }
            data.append(new_datum)



    # for d in data:
    #     assert 'task' in d

    return data

# Added for the extension to multi image
def extend_for_multi_image(args, data, task_to_examples):
    ## New code goes here, we get more examples for the context.
    multi_image_data = []
    for index, datum in tqdm(enumerate(data), total=len(data)):
        #   context_indexes = random.sample([elem for elem in task_to_examples[datum['task']] if elem != index], n_context_images)
        n_context_images = random.randint(1, args.max_context)
        context_indexes = random.sample(task_to_examples[datum['task']], n_context_images)
        # It can happen (expect to be very few times) that we sample the actual index, put it last in this case. 
        if index not in context_indexes:
            context_indexes.append(index)
        else:
            context_indexes.remove(index)
            context_indexes.append(index)

        multi_image_data.append({'uid': datum['uid'],
                                'img_id': [data[h_index]['img_id'] for h_index in context_indexes],
                                'img_source': [data[h_index]['img_source'] for h_index in context_indexes],
                                'sent': [data[h_index]['sent'] for h_index in context_indexes],
                                'text_source': [data[h_index]['text_source'] for h_index in context_indexes],
                                'task': datum['task'],
                                'label': [data[h_index]['label'] for h_index in context_indexes]})
    
    return multi_image_data


class PretrainDataset(Dataset):
    def __init__(self, split='vg', rank=-1, topk=-1, verbose=True, args=None, is_train=True, random_indexes=True, random_order_appearance=True):

        self.topk = topk
        self.verbose = verbose
        self.args = args

        self.max_context = args.max_context
        self.random_indexes = random_indexes
        self.random_order_appearance = random_order_appearance

        # Loading datasets to data
        self.sources = split.split(',')
        if self.verbose:
            print('Data sources: ', self.sources)

        # Answer Table from LXMERT (Could be removed)
        self.answer_table = AnswerTable()
        if self.verbose:
            print("Load an answer table of size %d." % (len(self.answer_table.ans2id_map())))

        self.img_ids_to_source = {}

        losses = args.losses.split(',')
        print(losses)
        data = []
        for img_source in self.sources:
            data_info_path = dataset_dir.joinpath(f'lxmert/{img_source}.json')
            with open(data_info_path) as f:
                _data = json.load(f)
                if self.verbose:
                    print(f"Loaded {len(_data)} data from", img_source)
                # source_img_ids.append([d['img_id'] for d in _data])
                for datum in _data:
                    self.img_ids_to_source[datum['img_id']] = img_source
                    # datum['img_source'] = img_source
                    datum['args'] = args
                    datum['is_train'] = is_train
                    datum['caption_only'] = args.caption_only

                    datum['lm'] = 'lm' in losses
                    datum['qa'] = 'qa' in losses
                    datum['ig'] = 'ig' in losses
                    datum['og'] = 'og' in losses
                    datum['cr'] = 'cr' in losses
                    datum['ci'] = 'ci' in losses

                    datum['caption'] = 'caption' in losses


                    datum['backbone'] = self.args.backbone

                data.extend(_data)


        # Modify the answers
        if 'qa' in args.losses:
            for datum in data:
                labelf = datum['labelf']
                for _qa_source, labels in labelf.items():
                    for label in labels:
                        for ans in list(label.keys()):
                            new_ans = self.answer_table.convert_ans(ans)
                            if self.answer_table.used(new_ans):
                                if ans != new_ans:
                                    label[new_ans] = label.pop(ans)
                            else:
                                label.pop(ans)


        if self.verbose:
            print("# images:", len(data))

        if self.topk > 0:
            data = data[:self.topk]
            if self.verbose:
                print(f"Use only {self.topk} data")

        if 'qa' in args.losses:
            self.evaluator = QAEvaluator(data)

        with Pool(8) as pool:
            if self.verbose:
                data = [datum for _data in tqdm(
                    pool.imap(get_datum, data), total=len(data), ncols=100, desc="Creating pretraining data examples") for datum in _data]
            else:
                data = [datum for _data in pool.imap(
                    get_datum, data) for datum in _data]


        if self.args.itm_cocoonly:
            caption_sources = ['mscoco']
        else:
            caption_sources = ['mscoco', 'vg']
        self.data_captions = [datum for datum in data if datum['text_source'] in caption_sources]
        self.n_data_captions = len(self.data_captions)
        self.task_to_examples = defaultdict(list)

        from collections import Counter
        task_counter = Counter()
        for index, datum in enumerate(data):
            try:
                task_counter.update([datum['task']])
                self.task_to_examples[datum['task']].append(index)
            except KeyError:
                print(datum)
                exit()
        if self.verbose:                
            print(task_counter)
        for k, v in task_counter.items():
            print(k, f'{v/len(data)*100:.1f}%')

        if self.verbose:
            print('# itm data:', self.n_data_captions)

        
        # Added for the extension to multi image, we do it here for minimal editing of original code
        data =  extend_for_multi_image(args, data, self.task_to_examples)

        self.data = data
        self.n_data = len(self.data)



        if self.verbose:
            print("# examples:", len(data))

        self.source_to_h5 = {
            'mscoco_resplit_train_train2014': h5py.File(coco_dir.joinpath('features').joinpath('train2014_obj36.h5')),
            'mscoco_resplit_train_val2014': h5py.File(coco_dir.joinpath('features').joinpath('val2014_obj36.h5')),
            'mscoco_resplit_val': h5py.File(coco_dir.joinpath('features').joinpath('resplit_val_obj36.h5')),
            'vgnococo': h5py.File(vg_dir.joinpath('features').joinpath('vg_gqa_obj36.h5')),

        }

        self.n_boxes = args.n_boxes

        if 't5' in self.args.backbone:
                self.tokenizer = T5Tokenizer.from_pretrained(
                    args.backbone)
        elif 'bart' in self.args.backbone:
            self.tokenizer = BartTokenizer.from_pretrained(args.backbone)

        additional_special_tokens = [f'<extra_id_{i}>' for i in range(100-1, -1, -1)] + \
                [f'<vis_extra_id_{i}>' for i in range(100-1, -1, -1)] + \
                [f'<img_extra_id_{i}>' for i in range(100-1, -1, -1)]
        special_tokens_dict = {'additional_special_tokens': additional_special_tokens}
        self.tokenizer.add_special_tokens(special_tokens_dict)


    def __len__(self):
        # return len(self.data)
        return self.n_data

    def _generate_text_(self, sents, img_ids):
        out_sentence = ""
        for i, sent in enumerate(sents):
            out_sentence += f'<img_extra_id_{img_ids[i]}> {sent} '
        return out_sentence.strip()

    def _generate_text_qa_(self, sents, answers, img_ids):
        out_sentence = ""
        for i, (sent, answer) in enumerate(zip(sents, answers)):
            out_sentence += f'<img_extra_id_{img_ids[i]}> {sent} {answer} '
        out_sentence += f'<img_extra_id_{img_ids[-1]}> {sents[-1]}'
        return out_sentence

    def _generate_img_ids_(self, n):
        img_ids = [i for i in range(self.args.max_context + 1)]
        if self.random_indexes:
            random.shuffle(img_ids)

        return img_ids[:n]

    def _generate_obj_ids_(self, img_ids):
        obj_ids = []
        for img_id in img_ids:
            for i in range(self.args.n_boxes):
                obj_ids.append(img_id * self.args.n_boxes + i)
        return obj_ids


    def __getitem__(self, idx):
        out_dict = {}
        out_dict['args'] = self.args
        
        datum = self.data[idx]
        uid = datum['uid']
        out_dict['uid'] = uid

        ###### Image ######
        img_ids = datum['img_id']
        sources = datum['img_source']

        '''
        fs = { f: self.source_to_h5[f] for f in sources}
        
        for f in fs:
            if isinstance(f, Path):
                path = self.source_to_h5[source]
                f = h5py.File(path, 'r')
                self.source_to_h5[source] = f
        '''

        # TODO! If we want to use oscar tags code has to be updated 

        if 't5' in self.args.backbone:

            text_source = datum['text_source']
            task = datum['task']

            loss_weight = 1

            # T5 Corrupt span
            if task == 'lm':
                assert text_source[-1] in ["mscoco", 'vg']

                prefix = "span prediction:"
                sents = datum['sent']
                min_index = 0
                source_texts = []
                target_texts = []
                for sent in sents:
                    source_text, target_text, min_index = preprocess.corrupt_spans(
                        sent, mask_ratio=self.args.word_mask_rate, prefix=prefix, min_index = min_index)

                    if self.args.oscar_tags:
                        input_tokens = [source_text]
                        input_tokens.append('tags:')
                        obj_ids = f[f'{img_id}/obj_id'][()]
                        for obj_id in obj_ids:
                            obj = vg_classes[obj_id]
                            if obj not in input_tokens:
                                input_tokens.append(obj)
                        source_text = ' '.join(input_tokens)

                    source_texts.append(source_text)
                    target_texts.append(target_text)

                img_indexes = self._generate_img_ids_(len(img_ids))
                obj_indexes = self._generate_obj_ids_(img_indexes)
                source_text = self._generate_text_(source_texts, img_indexes)
                target_text = ' '.join(target_texts)

            elif task == 'qa':
                assert text_source[-1] in ['vqa', 'gqa', 'visual7w'], (text_source, uid)

                labels = datum['label']
                answers = []
                for label in labels:
                    assert len(label) > 0

                    keys, values = zip(*label.items())
                    ans = ''
                    # single answer
                    if len(keys) == 1:
                        ans = keys[0]
                    # multiple answers -> sample one answer
                    else:
                        value_sum = sum(values)
                        prob = [value / value_sum for value in values]
                        choice = np.random.multinomial(1, prob).argmax()
                        ans = keys[choice]
                    answers.append(ans)

                sents = datum['sent']
                source_texts = []
                for i, sent in enumerate(sents):
                    if self.args.oscar_tags:
                        input_tokens = [source_text]
                        input_tokens.append('tags:')
                        obj_ids = self.source_to_h5[sources[i]][f'{img_ids[i]}/obj_id'][()]
                        for obj_id in obj_ids:
                            obj = vg_classes[obj_id]
                            if obj not in input_tokens:
                                input_tokens.append(obj)
                        sent = sent + ' ' + ' '.join(input_tokens)
                    source_texts.append(sent)
                img_indexes = self._generate_img_ids_(len(img_ids))
                obj_indexes = self._generate_obj_ids_(img_indexes)
                source_text = self._generate_text_qa_(source_texts, answers[:-1], img_indexes)

                if self.args.single_vqa_prefix:
                    source_text = f"vqa: {source_text}"
                else:
                    source_text = f"{text_source[-1]}: {source_text}"

                target_text = answers[-1]

            elif task == 'ig':
                assert text_source[-1] in ["mscoco", 'vg']
                sent = datum['sent'][-1]
                
                prefix = "image grounding:"
                source_text = f"{prefix} {sent}"

                if self.args.oscar_tags:
                    input_tokens = [source_text]
                    input_tokens.append('tags:')
                    obj_ids = self.source_to_h5[sources[0]][f'{img_ids[0]}/obj_id'][()]
                    for obj_id in obj_ids:
                        obj = vg_classes[obj_id]
                        if obj not in input_tokens:
                            input_tokens.append(obj)
                    source_text = ' '.join(input_tokens)

                # Target is always the last image
                img_indexes = self._generate_img_ids_(len(img_ids))                
                obj_indexes = self._generate_obj_ids_(img_indexes)
                                                

                target_text = f'<img_extra_id_{img_indexes[-1]}>' 
           

            if task == 'og':
                obj_ids = []
                attr_ids = [] 
                # We want to predict an object from the last image always. 
                obj_ids += self.source_to_h5[sources[-1]][f'{img_ids[-1]}/obj_id'][()].tolist()
                attr_ids += self.source_to_h5[sources[-1]][f'{img_ids[-1]}/attr_id'][()].tolist()

                captions = []
                for obj_id, attr_id in zip(obj_ids, attr_ids):
                    obj = vg_classes[obj_id]
                    attr = vg_attrs[attr_id]

                    caption = f'{attr} {obj}'
                    captions.append(caption)

                prefix = "object grounding:"

                img_indexes = self._generate_img_ids_(len(img_ids)) 
                obj_indexes = self._generate_obj_ids_(img_indexes)
                source_text, target_text = preprocess.refer_expression(
                    captions, obj_indexes[:self.args.n_boxes], self.args.n_ground, prefix=prefix, sort=False)

                sent = source_text

                loss_weight = self.args.ground_weight

            if task == 'cr':
                obj_ids = []
                attr_ids = [] 
                # We want to predict an object from the last image always. 
                obj_ids += self.source_to_h5[sources[-1]][f'{img_ids[-1]}/obj_id'][()].tolist()
                attr_ids += self.source_to_h5[sources[-1]][f'{img_ids[-1]}/attr_id'][()].tolist()

                captions = []
                for obj_id, attr_id in zip(obj_ids, attr_ids):
                    obj = vg_classes[obj_id]
                    attr = vg_attrs[attr_id]

                    caption = f'{attr} {obj}'
                    captions.append(caption)

                # prefix = "describe visual inputs:"
                prefix = "caption region:"
                img_indexes = self._generate_img_ids_(len(img_ids)) 
                obj_indexes = self._generate_obj_ids_(img_indexes)
                source_text, target_text = preprocess.ground_caption(
                    captions, obj_indexes[:self.args.n_boxes], self.args.n_ground, prefix=prefix, sort=False)

                sent = source_text
                loss_weight = self.args.ground_weight
                
            if task == 'ci':
                assert text_source[-1] in ["mscoco", 'vg']
                sent = datum['sent'][-1]
                # Target is always the first image
                img_indexes = self._generate_img_ids_(len(img_ids))  
                obj_indexes = self._generate_obj_ids_(img_indexes)
                prefix = "caption image:"
                source_text = f"{prefix} <img_extra_id_{img_indexes[-1]}>"
                target_text = sent

            '''
            # Random order for image features, we want to actually learn a mapping
            This is not required because image features do not have positions. The order of the features does not matter. 
            if self.random_order_appearance:
                random_indexes = [i for i in range(len(img_ids))]
                random.shuffle(random_indexes)
                img_ids = [img_ids[i] for i in random_indexes]                
                sources = [sources[i] for i in random_indexes]
                img_indexes = [img_indexes[i] for i in random_indexes]     
            '''

            input_ids = self.tokenizer.encode(
                source_text, padding=True, truncation=True, max_length=self.args.max_text_length)
            target_ids = self.tokenizer.encode(
                target_text, padding=True, truncation=True, max_length=self.args.gen_max_length)

            # if task in ['refer', 'itm']:
            #     target_ids = target_ids[:-1]

            out_dict['input_ids'] = torch.LongTensor(input_ids)
            out_dict['input_length'] = len(input_ids)
            out_dict['target_ids'] = torch.LongTensor(target_ids)
            out_dict['target_length'] = len(target_ids)

            out_dict['source_text'] = source_text
            out_dict['target_text'] = target_text

            out_dict['task'] = task
            out_dict['sent'] = sent

            out_dict['loss_weight'] = loss_weight


            feats = np.zeros(shape=(len(img_ids), self.n_boxes, 2048), dtype=np.float32)
            boxes_list = []
            try:
                for i, img_id in enumerate(img_ids):
                    # Get the top n_boxes taking prob into account
                    indexes = (-self.source_to_h5[sources[i]][f'{img_id}/attr_conf'][()]).argsort()[:self.n_boxes]
                    feats[i] = self.source_to_h5[sources[i]][f'{img_id}/features'][()][indexes]
                    # Normalize the boxes (to 0 ~ 1)
                    img_h = self.source_to_h5[sources[i]][f'{img_id}/img_h'][()]
                    img_w = self.source_to_h5[sources[i]][f'{img_id}/img_w'][()]
                    boxes = self.source_to_h5[sources[i]][f'{img_id}/boxes'][()][indexes]  # (x1, y1, x2, y2)
                    boxes[:, (0, 2)] /= img_w
                    boxes[:, (1, 3)] /= img_h
                    np.testing.assert_array_less(boxes, 1+1e-5)
                    # np.testing.assert_array_less(boxes, 1+5e-2)
                    np.testing.assert_array_less(-boxes, 0+1e-5)
                    boxes = torch.from_numpy(boxes)
                    boxes.clamp_(min=0.0, max=1.0)
                    boxes_list.append(boxes)
            except KeyError:
                print(uid)
                print(sources)
                print(img_id)
                exit()

            feats = torch.from_numpy(feats)
            out_dict['vis_feats'] = feats

            img_indexes = self.tokenizer.encode([f'<img_extra_id_{index}>' for index in img_indexes], add_special_tokens = False)
            obj_indexes = self.tokenizer.encode([f'<vis_extra_id_{index}>' for index in obj_indexes], add_special_tokens = False)

            out_dict['context_imgs_length'] = len(img_indexes)
            boxes = np.stack(boxes_list)
            boxes = torch.from_numpy(boxes)
            
            out_dict['boxes'] = boxes
            out_dict['img_indexes'] = img_indexes
            out_dict['obj_indexes'] = obj_indexes
            
            return out_dict

        elif 'bart' in self.args.backbone:

            text_source = datum['text_source']
            task = datum['task']

            loss_weight = 1

            # T5 Corrupt span
            if task == 'lm':
                assert text_source in ["mscoco", 'vg'], (datum, text_source)

                # LM only
                if self.args.losses == 'lm':
                    prefix = None
                else:
                    prefix = "denoise text:"
                sent = datum['sent']
                source_text, target_text = preprocess.corrupt_bart(
                    sent, mask_ratio=self.args.word_mask_rate, prefix=prefix)

                if self.args.oscar_tags:
                    input_tokens = [source_text]
                    input_tokens.append('tags:')
                    obj_ids = f[f'{img_id}/obj_id'][()]
                    for obj_id in obj_ids:
                        obj = vg_classes[obj_id]
                        if obj not in input_tokens:
                            input_tokens.append(obj)
                    source_text = ' '.join(input_tokens)

            elif task == 'qa':
                assert text_source in ['vqa', 'gqa',
                    'visual7w'], (text_source, uid)

                label = datum['label']
                assert len(label) > 0
                # for ans in list(label.keys()):
                #     label[self.answer_table.ans2id(ans)] = label.pop(ans)
                keys, values = zip(*label.items())
                # single answer
                if len(keys) == 1:
                    ans = keys[0]
                # multiple answers -> sample one answer
                else:
                    value_sum = sum(values)
                    prob = [value / value_sum for value in values]
                    choice = np.random.multinomial(1, prob).argmax()
                    ans = keys[choice]

                sent = datum['sent']

                if self.args.single_vqa_prefix:
                    source_text = f"vqa: {sent}"
                else:
                    source_text = f"{text_source}: {sent}"
                if self.args.oscar_tags:
                    input_tokens = [source_text]
                    input_tokens.append('tags:')
                    obj_ids = f[f'{img_id}/obj_id'][()]
                    for obj_id in obj_ids:
                        obj = vg_classes[obj_id]
                        if obj not in input_tokens:
                            input_tokens.append(obj)
                    source_text = ' '.join(input_tokens)
                target_text = ans

            elif task == 'itm':

                assert text_source in ["mscoco", 'vg']
                is_matched = 1
                sent = datum['sent']
                if random.random() < 0.5:
                    is_matched = 0

                    rand_idx = random.randint(0, self.n_data_captions-1)
                    # rand_idx = int(self.n_data_captions * random.random())

                    other_datum = self.data_captions[rand_idx]
                    # other_datum = self.data[random.randint(0, len(self.data)-1)]
                    while other_datum['img_id'] == img_id:

                        rand_idx = random.randint(
                            0, self.n_data_captions-1)
                        # rand_idx = int(self.n_data_captions * random.random())

                        other_datum = self.data_captions[rand_idx]
                        # other_datum = self.data[random.randint(0, len(self.data)-1)]
                    sent = other_datum['sent']

                prefix = "image text match:"
                source_text = f"{prefix} {sent}"

                if self.args.oscar_tags:
                    input_tokens = [source_text]
                    input_tokens.append('tags:')
                    obj_ids = f[f'{img_id}/obj_id'][()]
                    for obj_id in obj_ids:
                        obj = vg_classes[obj_id]
                        if obj not in input_tokens:
                            input_tokens.append(obj)
                    source_text = ' '.join(input_tokens)
                if is_matched:
                    target_text = 'true'
                else:
                    target_text = 'false'

            if task == 'ground_caption':
                obj_ids = f[f'{img_id}/obj_id'][()]
                attr_ids = f[f'{img_id}/attr_id'][()]

                captions = []
                for obj_id, attr_id in zip(obj_ids, attr_ids):
                    obj = vg_classes[obj_id]
                    attr = vg_attrs[attr_id]

                    caption = f'{attr} {obj}'
                    captions.append(caption)

                # prefix = "describe visual inputs:"
                prefix = "caption region:"
                source_text, target_text = preprocess.ground_caption(
                    captions, self.args.n_ground, prefix=prefix, sort=False)

                sent = source_text

                loss_weight = self.args.ground_weight

            if task == 'refer':
                obj_ids = f[f'{img_id}/obj_id'][()]
                attr_ids = f[f'{img_id}/attr_id'][()]

                captions = []
                for obj_id, attr_id in zip(obj_ids, attr_ids):
                    obj = vg_classes[obj_id]
                    attr = vg_attrs[attr_id]

                    caption = f'{attr} {obj}'
                    captions.append(caption)

                # prefix = "refer expressions:"
                prefix = "visual grounding:"
                source_text, target_text = preprocess.refer_expression(
                    captions, self.args.n_ground, prefix=prefix, sort=False)

                sent = source_text

                loss_weight = self.args.ground_weight

            input_ids = self.tokenizer.encode(
                source_text, padding=True, truncation=True, max_length=self.args.max_text_length)
            target_ids = self.tokenizer.encode(
                target_text, padding=True, truncation=True, max_length=self.args.gen_max_length)

            
            # if task in ['refer', 'itm']:
            #     target_ids = target_ids[:-1]

            out_dict['input_ids'] = torch.LongTensor(input_ids)
            out_dict['input_length'] = len(input_ids)
            out_dict['target_ids'] = torch.LongTensor(target_ids)
            out_dict['target_length'] = len(target_ids)

            out_dict['context_imgs_length'] = len(img_ids)

            out_dict['source_text'] = source_text
            out_dict['target_text'] = target_text

            out_dict['task'] = task
            out_dict['sent'] = sent

            out_dict['loss_weight'] = loss_weight

            feats = np.zeros(shape=(self.n_boxes, 2048), dtype=np.float32)
            try:
                f[f'{img_id}/features'].read_direct(feats)
            except KeyError:
                print(uid)
                print(sources)
                print(img_id)
                exit()

            feats = torch.from_numpy(feats)
            out_dict['vis_feats'] = feats

            # Normalize the boxes (to 0 ~ 1)
            img_h = f[f'{img_id}/img_h'][()]
            img_w = f[f'{img_id}/img_w'][()]
            boxes = f[f'{img_id}/boxes'][()]  # (x1, y1, x2, y2)
            boxes[:, (0, 2)] /= img_w
            boxes[:, (1, 3)] /= img_h
            np.testing.assert_array_less(boxes, 1+1e-5)
            # np.testing.assert_array_less(boxes, 1+5e-2)
            np.testing.assert_array_less(-boxes, 0+1e-5)
            boxes = torch.from_numpy(boxes)
            boxes.clamp_(min=0.0, max=1.0)
            out_dict['boxes'] = boxes

            return out_dict


    def collate_fn(self, batch):
        batch_entry = {}
        B = len(batch)

        args = self.args

        _, n_boxes, feat_dim = batch[0]['vis_feats'].shape
        
        S_W_L = max(entry['input_length'] for entry in batch)
        T_W_L = max(entry['target_length'] for entry in batch)

        max_context_images = max(entry['context_imgs_length'] for entry in batch)

        input_ids = torch.ones(B, S_W_L, dtype=torch.long) * self.tokenizer.pad_token_id
        target_ids = torch.ones(B, T_W_L, dtype=torch.long) * self.tokenizer.pad_token_id

        boxes = torch.zeros(B, max_context_images, n_boxes, 4, dtype=torch.float)
        vis_feats = torch.zeros(B, max_context_images, n_boxes, feat_dim, dtype=torch.float)

        img_indexes = torch.zeros((B, max_context_images * n_boxes), dtype=torch.long)
        obj_indexes = torch.zeros((B, max_context_images * n_boxes), dtype=torch.long)
        vis_attention = torch.zeros((B, max_context_images * n_boxes), dtype=torch.float)

        loss_weights = torch.ones(B, dtype=torch.float)

        sentences = []
        ans = []
        uids = []
        tasks = []

        source_text = []
        target_text = []

        for i, entry in enumerate(batch):
            input_ids[i, :entry['input_length']] = entry['input_ids']
            target_ids[i, :entry['target_length']] = entry['target_ids']

            boxes[i,:entry['context_imgs_length']] += entry['boxes']
            vis_feats[i,:entry['context_imgs_length']] += entry['vis_feats']

            if 'ans' in entry:
                ans.append(entry['ans'])

            if 'task' in entry:
                tasks.append(entry['task'])

            sentences.append(entry['sent'])
            uids.append(entry['uid'])

            img_indexes[i, :entry['context_imgs_length'] * n_boxes] += torch.LongTensor(np.repeat(entry['img_indexes'], n_boxes))
            obj_indexes[i, :entry['context_imgs_length'] * n_boxes] += torch.LongTensor(entry['obj_indexes'])
            vis_attention[i, :entry['context_imgs_length'] * n_boxes] += torch.ones((entry['context_imgs_length'] * n_boxes))

            if 'source_text' in entry:
                source_text.append(entry['source_text'])
            if 'target_text' in entry:
                target_text.append(entry['target_text'])

            if 'loss_weight' in entry:
                loss_weights[i] = entry['loss_weight']


        assert 't5' in args.backbone or 'bart' in args.backbone
        word_mask = target_ids != self.tokenizer.pad_token_id
        target_ids[~word_mask] = -100
        batch_entry['task'] = tasks

        batch_entry['source_text'] = source_text
        batch_entry['target_text'] = target_text

        batch_entry['input_ids'] = input_ids
        batch_entry['target_ids'] = target_ids

        batch_entry['boxes'] = boxes
        batch_entry['vis_feats'] = vis_feats

        batch_entry['loss_weights'] = loss_weights

        batch_entry['uid'] = uids
        batch_entry['sent'] = sentences

        batch_entry['img_indexes'] = img_indexes
        batch_entry['obj_indexes'] = obj_indexes
        batch_entry['vis_attention'] = vis_attention

        return batch_entry


def get_loader(args, split='vgnococo', mode='train',
               batch_size=32, workers=4, distributed=False, gpu=0,
               topk=-1):


    verbose = (gpu == 0)
    dataset = PretrainDataset(
        split,
        rank=gpu,
        topk=topk,
        verbose=verbose,
        args=args,
        is_train=(mode == 'train'),
        )

    if distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None

    
    if mode == 'train':
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(sampler is None),
            num_workers=workers, pin_memory=True, sampler=sampler,
            collate_fn=dataset.collate_fn)
    else:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers, pin_memory=True,
            sampler=sampler,
            shuffle=None if (sampler is not None) else False,
            collate_fn=dataset.collate_fn,
            drop_last=False)

    return loader


class QAEvaluator:
    def __init__(self, data):

        # Create QA Eval Data
        self.data = []
        for datum in data:
            sentf = datum['sentf']
            for sents_cat, sents in sentf.items():
                if sents_cat in datum['labelf']:    # A labeled dataset
                    labels = datum['labelf'][sents_cat]
                    for sent_idx, sent in enumerate(sents):
                        new_datum = {
                            'uid': make_uid(datum['img_id'], sents_cat, sent_idx),
                            'img_id': datum['img_id'],
                            'sent': sent,
                            'dset': sents_cat,
                            'label': labels[sent_idx]
                        }
                        self.data.append(new_datum)

        # uid2datum
        self.uid2datum = {}
        for datum in self.data:
            self.uid2datum[datum['uid']] = datum

    def evaluate(self, uid2ans: dict):
        score = 0.
        cnt = 0
        dset2score = defaultdict(lambda: 0.)
        dset2cnt = defaultdict(lambda: 0)
        for uid, ans in uid2ans.items():
            if uid not in self.uid2datum:   # Not a labeled data
                continue
            datum = self.uid2datum[uid]
            label = datum['label']
            dset = datum['dset']
            if ans in label:
                score += label[ans]
                dset2score[dset] += label[ans]
            cnt += 1
            dset2cnt[dset] += 1
        return dset2score, dset2cnt, score, cnt

    def _evaluate(self, uid2ans: dict, pprint=False):
        score = 0.
        cnt = 0
        dset2score = defaultdict(lambda: 0.)
        dset2cnt = defaultdict(lambda: 0)
        for uid, ans in uid2ans.items():
            if uid not in self.uid2datum:   # Not a labeled data
                continue
            datum = self.uid2datum[uid]
            label = datum['label']
            dset = datum['dset']
            if ans in label:
                score += label[ans]
                dset2score[dset] += label[ans]
            cnt += 1
            dset2cnt[dset] += 1
        accu = score / cnt
        dset2accu = {}
        for dset in dset2cnt:
            dset2accu[dset] = dset2score[dset] / dset2cnt[dset]

        if pprint:
            accu_str = "Overall Accu %0.4f, " % (accu)
            sorted_keys = sorted(dset2accu.keys())
            for key in sorted_keys:
                accu_str += "%s Accu %0.4f, " % (key, dset2accu[key])
            print(accu_str)

        return accu, dset2accu

    def dump_result(self, uid2ans: dict, path):
        raise NotImplementedError
