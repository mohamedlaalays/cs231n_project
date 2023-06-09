import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
# from transformers import SamModel, SamProcessor
# from PIL import Image
# from skimage import transform, io, segmentation
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide
# from torch.nn.utils.rnn import pad_sequence
# from torch.nn import functional as F
# import re
from datasets import NpzDataset
import random
import argparse
import json
import cv
from utils import *
from evaluation import *


def segment_data(data, exp_config):
    dataroot = f'dataset/npz_{args.model}/{data}/'
    dataset = NpzDataset(dataroot, exp_config)
    data_dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    dice_coefs, IoU_scores = [], []
    for i, batch in tqdm(enumerate(data_dataloader)):


        # if i == 1: break

        batched_output = sam([batch], multimask_output=False)

        # print(f'{batched_output=}')

        img_num = batch['img_num'].item()

        img_path = f"dataset/original/{data}/images/{data}_{img_num}.png"
        label_path = f"dataset/original/{data}/labels/{data}_{img_num}.png"
        label = io.imread(label_path)
        pred_mask = batched_output[0]['masks'][0].squeeze().numpy()
    

        dice_coef = compute_dice_coefficient(label, pred_mask)
        IoU_score = IoU(label, pred_mask)
        dice_coefs.append(dice_coef)
        IoU_scores.append(IoU_score)

        # randomly pick few images to qualitatively check the model performance
        # the images are saved in sample_images folder
        if random.randint(1, 100) < 100 and dice_coef < 0.5:
            input_points = batch["original_coords"].numpy()[0]
            input_labels = batch["point_labels"].numpy()[0]
            original_box = batch["original_box"].numpy()[0]
            superposed_image = superpose_img_label(img_path, label_path, img_num)
            show_points_and_boxes_gt_on_image(superposed_image, [original_box], input_points, input_labels)
            superpose_img_mask(img_path, label_path, pred_mask, img_num, exp_config['bbox_size'])
            print(f'{img_num}: {dice_coef=}')


        # print(f'{dice_coef=}')
        
        # break


    avg_dice_coef = sum(dice_coefs) / len(dice_coefs)
    avg_iou_score = sum(IoU_scores) / len(IoU_scores)
    print(f'The average dice coefficient of {data} dataset with {exp_config} configuration: {avg_dice_coef}')
    print(f'The intersection over union (IoU) of {data} dataset with {exp_config} configuration: {avg_iou_score}')
    return avg_dice_coef, avg_iou_score


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--dataset", type=str, required=True,
                        help='Choose one of the two dataset',
                        choices=('malignant', 'benign'))
    parser.add_argument("--model", type=str, required=True,
                        help='Currently supports the huge and the base models',
                        choices=("base", "huge"))
    parser.add_argument("--use_cpu", action='store_true')

    args = parser.parse_args()
    return args



if __name__ == "__main__":

    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    models = {
        'huge': 'sam_vit_h_4b8939.pth',
        'large': 'sam_vit_l_0b3195.pth',
        'base': 'sam_vit_b_01ec64.pth'
    }

    model_types = {
        'huge': 'vit_h',
        'large': 'vit_l',
        'base': 'vit_b'
    }

    model_type = model_types[args.model]
    model = models[args.model]
    device = "cuda" if not args.use_cpu else "cpu"
    sam_checkpoint = f"models/{model}"

    print(f'starting ....... {model=}')

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    resize_transform = ResizeLongestSide(sam.image_encoder.img_size)
    dataset = args.dataset

    results = {}
    exp_config = {
            'bbox_size': 20
        }
    
    # for num_pts in range(1, 10, 1):
    #     exp_config['num_pts'] = num_pts
    #     avg_dice_coef, avg_iou_score = segment_data(dataset, exp_config)
    #     results[num_pts] = {
    #                             'avg_dice_coef': avg_dice_coef,
    #                             'avg_iou_score': avg_iou_score
    #                           }
        
    


    # exp_config['num_pts'] = 4
    avg_dice_coef, avg_iou_score = segment_data(dataset, exp_config)

    results['baseline'] = {
                                'avg_dice_coef': avg_dice_coef,
                                'avg_iou_score': avg_iou_score
                              }

    # file_path = f'results/{dataset}_{model_type}_baseline.json'
    # with open(file_path, "w") as json_file:
    #     json.dump(results, json_file)
