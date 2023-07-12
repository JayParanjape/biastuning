import os
import torch
import PIL.Image as Image
import matplotlib.pyplot as plt
import sys

sys.path.append("/home/ubuntu/Desktop/Domain_Adaptation_Project/repos/biastuning/")

from utils import *

results_folder_name = 'endovis18_10label_textaffine_decdertuning_4e-4_adamw_focal_alpha75e-2_gamma_2_256_bs64_rsz_manyaug_blanklables'

ious_all = {}
for object in os.listdir(results_folder_name):
    ious = []
    print("Starting object: ", object)
    preds_path = os.path.join(results_folder_name, object, 'rescaled_preds')
    gt_path = os.path.join(results_folder_name, object, 'rescaled_gt')
    for i,im in enumerate(os.listdir(gt_path)):
        if i<13:
            continue
        label = np.array(Image.open(os.path.join(gt_path,im)))[60:306,150:400]
        label = (label>127)+0
        pred = np.array(Image.open(os.path.join(preds_path,im)))[60:306, 150:400]
        pred = (pred>127) + 0
        plt.imshow(label)
        plt.show()
        plt.imshow(label)
        plt.show()
        print(label.shape)
        print(pred.shape)
        print(np.unique(pred))
        1/0