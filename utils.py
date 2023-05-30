import numpy as np
import torch
def dice_coef(y_true, y_pred, smooth=1):
    # print(y_pred.shape, y_true.shape)
    intersection = torch.sum(y_true * y_pred,axis=(1,2))
    union = torch.sum(y_true, axis=(1,2)) + torch.sum(y_pred, axis=(1,2))
    dice = ((2. * intersection + smooth)/(union + smooth)).mean()
    # print(dice)
    return dice

def iou_coef(y_true, y_pred, smooth=1):
    intersection = torch.sum(torch.abs(y_true * y_pred),axis=(1,2))
    union = torch.sum(y_true,axis=(1,2))+torch.sum(y_pred,axis=(1,2))-intersection
    iou = ((intersection + smooth) / (union + smooth)).mean()
    return iou

def running_stats(y_true, y_pred, smooth = 1):
    intersection = torch.sum(y_true * y_pred,axis=(1,2))
    union = torch.sum(y_true, axis=(1,2)) + torch.sum(y_pred, axis=(1,2))
    return intersection, union

def dice_collated(running_intersection, running_union, smooth =1):
    dice = ((2. * running_intersection + smooth)/(running_union + smooth)).sum()
    return dice

def dice_batchwise(running_intersection, running_union, smooth =1):
    dice = ((2. * running_intersection + smooth)/(running_union + smooth))
    return dice