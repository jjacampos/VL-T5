import getopt
import json
import os
import bbox_visualizer as bbv

# import numpy as np
import sys
from collections import OrderedDict
from argparse import ArgumentParser
from tkinter import image_names
from click import Argument
import numpy as np
import torch
import cv2
import os
from visualizing_image import SingleImageViz

from modeling_frcnn import GeneralizedRCNN
from processing_image import Preprocess, _scale_box, _clip_box, _scale_box_ours
from utils import Config, get_data
from tqdm import tqdm

from IPython.display import clear_output, Image, display
import PIL.Image
import io

import pickle


URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/images/input.jpg"
OBJ_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/objects_vocab.txt"
ATTR_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/attributes_vocab.txt"

"""
USAGE:
``python extracting_data.py -i <img_dir> -o <dataset_file>.datasets <batch_size>``
"""

def get_bboxes_and_indexes(objects):

    bboxes = []
    indexes = []
    for object in objects:
        bbox = object['bbox']
        new_box = [bbox[0], bbox[1], bbox[0] + bbox[3], bbox[1] + bbox[2]]
        bboxes.append(new_box)
        indexes.append(object["index"])

    return torch.FloatTensor(bboxes), indexes

# for visualizing output
def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = io.BytesIO()
    PIL.Image.fromarray(a).save('./test.jpg')

def plot_bboxes(img, bboxes):
    img = bbv.draw_multiple_rectangles(img, bboxes)
    cv2.imwrite('./updated.jpg', img)

def get_bboxes2(objects):
    return torch.tensor([[169.2822, 214.2777, 485.0688, 469.7582],
         [100.7333, 251.5733, 394.4173, 480.0000],
         [586.0610,   0.0000, 638.9166, 353.4768],
         [608.1254,  61.8310, 637.5305, 423.0863],
         [281.5092, 162.0801, 328.3618, 194.3500],
         [157.5595,  93.0654, 392.0049, 461.4327],
         [222.4542, 291.4030, 565.9203, 480.0000],
         [ 41.3654, 181.6297, 349.8858, 450.4873],
         [ 13.9046,   1.7001, 207.7996, 437.1931],
         [381.4378, 310.5487, 618.7042, 417.9735]])

def main(args):

    objids = get_data(OBJ_URL) 
    
    attrids = get_data(ATTR_URL)

    frcnn_config = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
    if torch.cuda.is_available():
            frcnn_config.model.device = "cuda"
    frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_config)
    image_preprocess = Preprocess(frcnn_config)

    images_data = {}

    for img_file in tqdm(os.listdir(args.imgs_input_path), total=len(os.listdir(args.imgs_input_path))):
        full_img_path = os.path.join(args.imgs_input_path, img_file)

        """
        frcnn_visualizer = SingleImageViz(full_img_path, id2obj=objids, id2attr=attrids) 
        """        

        scene_graph_file = f"{img_file.split('.')[0]}_scene.json"
        try:
            scene_graph = json.load(open(os.path.join(args.scene_graphs_input_path, scene_graph_file)))
        except:
            print(f"Error with the following path: {scene_graph_file}")
            continue

        if len(scene_graph['scenes']) > 1:
            print(f'{full_img_path} has to be checked')

        try:
            images, sizes, scales_yx = image_preprocess(full_img_path)
        except:
            print(f"Error when processing {full_img_path}")
            continue
        bboxes, indexes = get_bboxes_and_indexes(scene_graph['scenes'][0]['objects'])
        bboxes = _scale_box_ours(bboxes, scales_yx)

        output_dict = frcnn(
            images,
            sizes,
            scales_yx=scales_yx,
            padding="max_detections",
            return_tensors="pt",
            gt_boxes=bboxes,
        )

        """
        frcnn_visualizer.draw_boxes(
            output_dict.get("boxes"),
            output_dict.get("obj_ids"),
            output_dict.get("obj_probs"),
            output_dict.get("attr_ids"), 
            output_dict.get("attr_probs"),
        ) 
        
        showarray(frcnn_visualizer._get_buffer())  
        """
        import pdb
        pdb.set_trace()   
        output_dict["indexes"] = indexes
        img_id = img_file.split('.png')[0]

        images_data[img_id] = output_dict
        
    pickle.dump(images_data, open(args.output_path, 'wb'))

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--imgs_input_path')
    parser.add_argument('--scene_graphs_input_path')
    parser.add_argument('--output_path')

    args = parser.parse_args()

    main(args)


