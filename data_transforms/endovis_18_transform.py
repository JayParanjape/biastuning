import random
import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import functional as F
from torch.nn.functional import pad


class ENDOVIS_18_Transform():
    def __init__(self, config):
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1,1,1)
        self.pixel_std = torch.Tensor([53.395, 57.12, 57.375]).view(-1,1,1)
        self.degree = config['data_transforms']['rotation_angle']
        self.saturation = config['data_transforms']['saturation']
        self.brightness = config['data_transforms']['brightness']
        self.img_size = config['data_transforms']['img_size']
        self.resize = transforms.Resize(self.img_size-1, max_size=self.img_size, antialias=True)

        self.data_transforms = config['data_transforms']

    def __call__(self, img, mask, apply_norm=True, is_train=True):
        if is_train:
            #flip horizontally with some probability
            if self.data_transforms['use_horizontal_flip']:
                p = random.random()
                if p<0.5:
                    img = F.hflip(img)
                    mask = F.hflip(mask)

            #rotate with p1 probability
            if self.data_transforms['use_rotation']:
                p = random.random()
                if p<0.5:
                    deg = 1+random.choice(list(range(self.degree)))
                    img = F.rotate(img, angle = deg)
                    mask = F.rotate(mask, angle=deg)

            #adjust saturation with some probability
            if self.data_transforms['use_saturation']:
                p = random.random()
                if p<0.2:
                    img = F.adjust_saturation(img, self.saturation)
            
            #adjust brightness with some probability
            if self.data_transforms['use_brightness']:
                p = random.random()
                if p<0.5:
                    img = F.adjust_brightness(img, self.brightness*random.random())

        #take random crops of img size X img_size such that label is non zero
        if self.data_transforms['use_random_crop']:
            fallback = 20
            fall_back_ctr = 0
            repeat_flag = True
            while(repeat_flag):
                fall_back_ctr += 1                    
                t = transforms.RandomCrop((self.img_size, self.img_size))
                i,j,h,w = t.get_params(img, (self.img_size, self.img_size))
                
                #if mask is all zeros, exit the loop
                if not mask.any():
                    repeat_flag = False
                
                #fallback to avoid long loops
                if fall_back_ctr >= fallback:
                    temp1, temp2, temp3 = np.where(mask!=0)
                    point_of_interest = random.choice(list(range(len(temp2))))
                    i = temp2[point_of_interest] - (h//2)
                    j = temp3[point_of_interest] - (w//2)
                    repeat_flag = False

                cropped_img = F.crop(img, i, j, h, w)
                cropped_mask = F.crop(mask, i, j, h, w)
                if cropped_mask.any():
                    repeat_flag = False
            img = cropped_img
            mask = cropped_mask
        else:
            #if no random crops then perform resizing
            b_min = 0
            img = self.resize(img)
            mask = self.resize(mask)
            #pad if necessary
            h, w = img.shape[-2:]
            padh = self.img_size - h
            padw = self.img_size - w
            img = pad(img, (0, padw, 0, padh), value=b_min)
            mask = pad(mask, (0, padw, 0, padh), value=b_min)


        #apply centering based on SAM's expected mean and variance
        if apply_norm:
            b_min=0
            #scale intensities to 0-255
            b_min,b_max = 0, 255
            img = (img - self.data_transforms['a_min']) / (self.data_transforms['a_max'] - self.data_transforms['a_min'])
            img = img * (b_max - b_min) + b_min
            img = torch.clamp(img,b_min,b_max)

            #center around SAM's expected mean
            img = (img - self.pixel_mean)/self.pixel_std
            
        return img, mask