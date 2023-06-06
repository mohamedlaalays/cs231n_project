import torch
from transformers import SamModel, SamProcessor
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from utils import *
import os

"""Compute soerensen-dice coefficient.

  compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
  and the predicted mask `mask_pred`. 
  
  Args:
    mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
    mask_pred: 3-dim Numpy array of type bool. The predicted mask.

  Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN
  """
# credit to: http://medicaldecathlon.com/files/Surface_distance_based_measures.ipynb
def compute_dice_coefficient(mask_gt, mask_pred):
    
    # print(f'{mask_gt.shape}=')
    # print(f'{mask_pred.shape}')
    volume_sum = mask_gt.sum() + mask_pred.sum()
    if volume_sum == 0:
        return np.NaN
    volume_intersect = (mask_gt & mask_pred).sum()
    return 2*volume_intersect / volume_sum



"""
IoU score of the predicted and the ground truth
"""
def IoU(mask_gt, mask_pred):
    intersection = np.logical_and(mask_gt, mask_pred)
    union = np.logical_or(mask_gt, mask_pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score
