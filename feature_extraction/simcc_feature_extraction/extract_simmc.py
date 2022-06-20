import getopt
import json
import os

# import numpy as np
import sys
from collections import OrderedDict
from argparse import ArgumentParser
from tkinter import image_names
from click import Argument
import numpy as np
import torch
import os
from visualizing_image import SingleImageViz

from modeling_frcnn import GeneralizedRCNN
from processing_image import Preprocess
from utils import Config
from tqdm import tqdm

import pickle

"""
USAGE:
``python extracting_data.py -i <img_dir> -o <dataset_file>.datasets <batch_size>``
"""

def main(args):

    frcnn_config = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
    if torch.cuda.is_available():
            frcnn_config.model.device = "cuda"
    frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_config)
    image_preprocess = Preprocess(frcnn_config)

    images_data = {}

    for img_file in tqdm(os.listdir(args.input_path), total=len(os.listdir(args.input_path))):
        try:
            full_img_path = os.path.join(args.input_path, img_file)
            images, sizes, scales_yx = image_preprocess(full_img_path)
            output_dict = frcnn(images, sizes, scales_yx=scales_yx, padding='max_detections', max_detections=frcnn_config.max_detections, return_tensors='pt')
            img_id = img_file.split('.png')[0]
            images_data[img_id] = output_dict
        except:
            print('Should not happen')
            continue
        
    pickle.dump(images_data, open(args.output_path, 'wb'))

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--input_path')
    parser.add_argument('--output_path')

    args = parser.parse_args()

    main(args)


