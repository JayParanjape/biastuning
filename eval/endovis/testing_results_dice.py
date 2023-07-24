import os
from PIL import Image
import sys
from matplotlib import pyplot as plt
import torch

sys.path.append("/home/ubuntu/Desktop/Domain_Adaptation_Project/repos/biastuning/")
from utils import *

test_path = "/home/ubuntu/Desktop/Domain_Adaptation_Project/repos/biastuning/eval/endovis/samzs_results"

#when not differentiating between the forceps, add mbp to the first tuple
instruments = [('lgr','rgr'),('llnd','rlnd'),('lpf','rpf')]
# instruments = [('Left Grasping Retractor','Right Grasping Retractor'),('Left Large Needle Driver','Right Large Needle Driver'),('Left Prograsp Forceps','Right Prograsp Forceps')]

for dataset in sorted(os.listdir(test_path)):
    for instrument in instruments:
        dices = []
        ious = []
        if len(instrument)==3:
            gt_path1 = os.path.join(test_path, dataset,instrument[0],'rescaled_gt')
            gt_path2 = os.path.join(test_path, dataset,instrument[2],'rescaled_gt')
            extra_preds_path = os.path.join(test_path, dataset,instrument[2],'rescaled_preds')
        else:
            gt_path = os.path.join(test_path, dataset,instrument[0],'rescaled_gt')
        left_preds_path = os.path.join(test_path, dataset,instrument[0],'rescaled_preds')
        right_preds_path = os.path.join(test_path, dataset,instrument[1],'rescaled_preds')
        for frame in sorted(os.listdir(left_preds_path)):
            if len(instrument)==3:
                gold1 = ((plt.imread(os.path.join(gt_path1,frame))[:,:,0][58:-52,143:-126])>=0.5)+0
                gold2 = ((plt.imread(os.path.join(gt_path2,frame))[:,:,0][58:-52,143:-126])>=0.5)+0
                extra_pred = ((plt.imread(os.path.join(extra_preds_path, frame))[:,:,0][58:-52,143:-126])>=0.5)
                gold = (gold1 | gold2)+0
            else:
                gold = ((plt.imread(os.path.join(gt_path,frame))[:,:,0][58:-52,143:-126])>=0.5)+0
            left_pred = ((plt.imread(os.path.join(left_preds_path, frame))[:,:,0][58:-52,143:-126])>=0.5)
            right_pred = ((plt.imread(os.path.join(right_preds_path, frame))[:,:,0][58:-52,143:-126])>=0.5)
            
            pred = (left_pred | right_pred)
            if len(instrument)==3:
                pred = (pred | extra_pred)
            pred = pred + 0
            gold = torch.Tensor(gold).unsqueeze(0)
            pred = torch.Tensor(pred).unsqueeze(0)
            dices.append(dice_coef(gold, pred))
            ious.append(iou_coef(gold, pred))

       
        # if instrument==('lpf','rpf') and dataset=='instrument_2':
        #     print(dices)
        #     print(os.path.join(left_preds_path, frame))
        #     plt.imshow(plt.imread(os.path.join(left_preds_path, frame)),cmap='gray')
        #     plt.imshow(pred[0],'gray')
        #     plt.show()
        #     plt.imshow(gold[0],cmap='gray')
        #     plt.show()
        #     1/0

        print(f"Dataset: {dataset}, instrument: {instrument}, dice: {torch.mean(torch.Tensor(dices))}, iou: {torch.mean(torch.Tensor(ious))}")
    print('\n')