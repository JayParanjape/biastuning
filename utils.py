import numpy as np
import torch
import torch.nn.functional as F
import argparse
import torch.nn as nn

def dice_coef(y_true, y_pred, smooth=1):
    # print(y_pred.shape, y_true.shape)
    intersection = torch.sum(y_true * y_pred,axis=(-1,-2))
    union = torch.sum(y_true, axis=(-1,-2)) + torch.sum(y_pred, axis=(-1,-2))
    dice = ((2. * intersection + smooth)/(union + smooth)).mean()
    # print(dice)
    return dice

def iou_coef(y_true, y_pred, smooth=1):
    intersection = torch.sum(torch.abs(y_true * y_pred),axis=(-1,-2))
    union = torch.sum(y_true,axis=(-1,-2))+torch.sum(y_pred,axis=(-1,-2))-intersection
    iou = ((intersection + smooth) / (union + smooth)).mean()
    return iou

def running_stats(y_true, y_pred, smooth = 1):
    intersection = torch.sum(y_true * y_pred,axis=(-1,-2))
    union = torch.sum(y_true, axis=(-1,-2)) + torch.sum(y_pred, axis=(-1,-2))
    return intersection, union

def dice_collated(running_intersection, running_union, smooth =1):
    if len(running_intersection.size())==2:
        dice = (torch.mean((2. * running_intersection + smooth)/(running_union + smooth),dim=1)).sum()
    else:
        dice = ((2. * running_intersection + smooth)/(running_union + smooth)).sum()
    return dice

def dice_batchwise(running_intersection, running_union, smooth =1):
    dice = ((2. * running_intersection + smooth)/(running_union + smooth))
    return dice

def dice_loss(y_pred, y_true):
    numerator = (2 * torch.sum(y_true * y_pred, axis=(-1,-2)))
    denominator = torch.sum(y_true + y_pred, axis=(-1,-2))

    return 1 - torch.mean((numerator+1) / (denominator+1))

def weighted_ce_loss(y_pred, y_true, alpha=64, smooth=1):
    weight1 = torch.sum(y_true==1,dim=(-1,-2))+smooth
    weight0 = torch.sum(y_true==0, dim=(-1,-2))+smooth
    multiplier_1 = weight0/(weight1*alpha)
    multiplier_1 = multiplier_1.view(-1,1,1)
    # print(multiplier_1.shape)
    # print(y_pred.shape)
    # print(y_true.shape)

    loss = -torch.mean(torch.mean((multiplier_1*y_true*torch.log(y_pred)) + (1-y_true)*(torch.log(1-y_pred)),dim=(-1,-2)))
    return loss

def focal_loss(y_pred, y_true, alpha_def=0.75, gamma=3):
    # print('going back to the default value of alpha')
    alpha = alpha_def
    ce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true, reduction="none")
    assert (ce_loss>=0).all()
    p_t = y_pred * y_true + (1 - y_pred) * (1 - y_true)
    # 1/0
    loss = ce_loss * ((1 - p_t) ** gamma)
    alpha_t = alpha * y_true + (1 - alpha) * (1 - y_true)
    loss = alpha_t * loss
    loss = torch.sum(loss, dim=(-1,-2))
    return loss.mean()

def multiclass_focal_loss(y_pred, y_true, alpha = 0.75, gamma=3):
    ce = y_true*(-torch.log(y_pred))
    weight = y_true * ((1-y_pred)**gamma)
    fl = torch.sum(alpha*weight*ce, dim=(-1,-2))
    return torch.mean(fl)

def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform"""