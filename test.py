import os
import sys
import numpy as np
from data_utils import *
from model import *
from utils import *
import torch

def test(config_data, config_model, pretrained_path, test_start, test_end, device='cuda:0'):
    with torch.no_grad():
            transform = Slice_Transforms(config_data)
            all_label_list = config_data['data']['label_list']
            all_label_names = config_data['data']['label_names']
            all_label_dict = {}
            for i,ln in enumerate(all_label_names):
                all_label_dict[ln] = i
            model = Prompt_Adapted_SAM(device=device,config=config_model,label_text_dict=all_label_dict)

            if pretrained_path is not None:
                state_dict = torch.load(pretrained_path)
                model.load_state_dict(state_dict, strict=True)

            #initialize dice scores for all labels
            dices = {}
            for l in all_label_names:
                dices[l] = []

            model = model.to(device)
            data_dir = config_data['data']['root_path']
            for name in os.listdir(data_dir+'/images'):
                print(name)
                #only test for val set
                if int(name[:name.find('.')])>=test_start and int(name[:name.find('.')])<test_end:
                    im_path = os.path.join(data_dir, 'images', name)
                    label_path = os.path.join(data_dir, 'labels', name)
                    im_ = nib.load(im_path)
                    mask_ = nib.load(label_path)
                    mask_ = np.asanyarray(mask_.dataobj)
                      
                    #image loading and conversion to rgb by replicating channels
                    if config_data['data']['volume_channel']==2: #data originally is HXWXC
                        im_ = (torch.Tensor(np.asanyarray(im_.dataobj)).permute(2,0,1).unsqueeze(1).repeat(1,3,1,1))
                        mask_ = torch.Tensor(mask_).permute(2,0,1)
                    else: #data originally is CXHXW
                        im_ = (torch.Tensor(np.asanyarray(im_.dataobj)).unsqueeze(1).repeat(1,3,1,1))
                    for i in range(0,im_.shape[0],8):
                        im = im_[i:i+8]
                        im = transform(im)
                        text_li = []                        
                        mask = mask_[i:i+8]

                        for num,l in enumerate(all_label_list):
                            mask_l = ((mask==l)+0)
                            mask_l = transform(mask_l, is_mask=True)
                            if config_model['img_type']=='ct':
                                text_li = ["computerized tomography of a " + all_label_names[num]]*im.shape[0]

                            outputs = model(im.to(device), text_li)
                            outputs = outputs>=0.5+0
                            dice_l = dice_coef(mask_l, outputs.cpu())
                            dices[all_label_names[num]].append(dice_l.numpy())

            #take the average dice score in each label
            for l in all_label_names:
                dices[l] = np.mean(dices[l])
            print(dices)