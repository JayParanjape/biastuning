import random
import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, TensorDataset
from torchvision import datasets, models
from torchvision import transforms
from torchvision.transforms import functional as F
from torch.nn.functional import pad
from skimage.transform import resize
import nibabel as nib
import time

from data_transforms.endovis_transform import ENDOVIS_Transform
from data_transforms.endovis_18_transform import ENDOVIS_18_Transform
from data_transforms.cholec_8k_transform import Cholec_8k_Transform
from data_transforms.ultrasound_transform import Ultrasound_Transform
from data_transforms.kvasirSeg_transform import kvasirSeg_Transform
from data_transforms.ChestXDet_transform import ChestXDet_Transform

class Slice_Transforms:
    def __init__(self, config=None):
        #SAM encoder expects images to be centered around tehe following mean and variance, how to change it for medical datasets?
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1,1,1).unsqueeze(0)
        self.pixel_std = torch.Tensor([53.395, 57.12, 57.375]).view(-1,1,1).unsqueeze(0)
        self.img_size = config['data_transforms']['img_size']
        self.resize = transforms.Resize(self.img_size-1, max_size=self.img_size, antialias=True)
        self.a_min = config['data_transforms']['a_min']
        self.a_max = config['data_transforms']['a_max']
        

    def __call__(self, image, is_mask=False, apply_mean_norm=True):
        # image = torch.Tensor(image)
        b_min=0
        if not is_mask:
            #scale intensities to 0-255
            b_min,b_max = 0, 255
            image = (image - self.a_min) / (self.a_max - self.a_min)
            image = image * (b_max - b_min) + b_min
            image = torch.clamp(image,b_min,b_max)

            #center around SAM's expected mean
            if apply_mean_norm:
                image = (image - self.pixel_mean)/self.pixel_std
        
        image = self.resize(image)
        #pad if necessary
        h, w = image.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        image = pad(image, (0, padw, 0, padh), value=b_min)
        return image


class Generic_Dataset_3d(Dataset):
    def __init__(self, config, is_train=False, folder_start=0, folder_end=40, shuffle_list=True):
        super().__init__()
        self.root_path = config['data']['root_path']
        self.img_path_list = []
        self.label_path_list = []
        self.label_names_text = []
        self.label_names = config['data']['label_names']
        self.label_list = config['data']['label_list']
        self.is_train = is_train
        self.folder_start = folder_start
        self.folder_end = folder_end
        self.config = config

        self.populate_lists()
        if shuffle_list:
            p = [x for x in range(len(self.img_path_list))]
            random.shuffle(p)
            self.img_path_list = [self.img_path_list[pi] for pi in p]
            self.label_path_list = [self.label_path_list[pi] for pi in p]
            self.label_names_text = [self.label_names_text[pi] for pi in p]


        #define data transforms
        self.transform = Slice_Transforms(config=config)

    def populate_lists(self):
        # print(self.folder_start, self.folder_end, self.label_list)

        for case_no in sorted(os.listdir(os.path.join(self.root_path,'images'))):
            case_idx = int(case_no[:case_no.find('.')])
            if not((case_idx>=self.folder_start) and (case_idx<self.folder_end)):
                continue
            im_path = os.path.join(self.root_path, 'images',case_no)
            label_path = os.path.join(self.root_path, 'labels', case_no)
            for i in range(len(self.label_list)):
                self.img_path_list.append(im_path)
                self.label_path_list.append(label_path)
                self.label_names_text.append(self.label_names[i])

    def __len__(self):
        assert(len(self.img_path_list)==len(self.label_path_list))
        return len(self.img_path_list)

    def __getitem__(self, index):
        #load masks and images
        im = nib.load(self.img_path_list[index])
        label_text = self.label_names_text[index]
        label_segmask_no = self.label_list[self.label_names.index(label_text)]
        mask = nib.load(self.label_path_list[index])
        mask = np.asanyarray(mask.dataobj)

        #convert general mask into prompted segmentation mask per according to label name
        gold = (mask==label_segmask_no)
        gold = torch.Tensor(gold+0)

        #convert to C, H, W
        if self.config['data']['volume_channel']==2:
            gold = gold.permute(2,0,1)

        # use gaussian with mean as the slice with biggest mask and a big variance
        mu, sigma = (torch.argmax(torch.sum(gold, dim=(1,2)))), self.config['data']['sampling_deviation'] # mean and standard deviation
        s = (np.random.normal(mu, sigma, self.config['data']['samples_per_slice'])).astype(int)
        s = [max(i,0) for i in s]
        s = [min(i,gold.shape[0]-2) for i in s]
        try:
            gold = gold[s]
            gold = self.transform(gold, is_mask=True)
        except:
            s = (np.random.normal(mu, sigma, self.config['data']['samples_per_slice'])).astype(int)
            s = [max(i,0) for i in s]
            s = [min(i,gold.shape[0]-2) for i in s]
            gold = gold[s]
            gold = self.transform(gold, is_mask=True)


        # plt.imshow(gold, cmap='gray')
        # plt.show()
        #convert all grayscale pixels due to resizing back to 0, 1
        gold = (gold>=0.5)+0
        # plt.imshow(gold, cmap='gray')
        # plt.show()
        #only consider some k slices at random
        
        
        #image loading and conversion to rgb by replicating channels
        if self.config['data']['volume_channel']==2: #data originally is HXWXC
            im = (torch.Tensor(np.asanyarray(im.dataobj)).permute(2,0,1).unsqueeze(1).repeat(1,3,1,1))[s]
        else: #data originally is CXHXW
            im = (torch.Tensor(np.asanyarray(im.dataobj)).unsqueeze(1).repeat(1,3,1,1))[s]
        im = self.transform(im)
        
        
        return im, gold, label_segmask_no, label_text


class IDRID_Transform():
    def __init__(self, config):
        self.pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(-1,1,1)
        self.pixel_std = torch.Tensor([53.395, 57.12, 57.375]).view(-1,1,1)
        self.degree = config['data_transforms']['rotation_angle']
        self.saturation = config['data_transforms']['saturation']
        self.brightness = config['data_transforms']['brightness']
        self.img_size = config['data_transforms']['img_size']
        self.resize = transforms.Resize(self.img_size-1, max_size=self.img_size, antialias=True)

        self.data_transforms = config['data_transforms']

    def __call__(self, img, mask, apply_norm, is_train):
        #crop the image so that only the main arrea is in consideration
        img = img[:,:,270:3700]
        mask = mask[:,:,270:3700]
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
                    img = F.rotate(img, angle = self.degree)
                    mask = F.rotate(mask, angle=self.degree)

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
            

class IDRID_Dataset(Dataset):
    def __init__(self, config, is_train=False, folder_start=0, folder_end=40, shuffle_list=True, apply_norm=True):
        super().__init__()
        self.root_path = config['data']['root_path']
        self.img_path_list = []
        self.label_path_list = []
        self.label_names_text = []
        self.label_names = config['data']['label_names']
        self.label_list = config['data']['label_list']
        self.is_train = is_train
        self.folder_start = folder_start
        self.folder_end = folder_end
        self.config = config
        self.apply_norm = apply_norm
        self.acronym = {
            'Microaneurysms': 'MA',
            'Haemorrhages': 'HE',
            'Hard Exudates': 'EX',
            'Optic Disc': 'OD',
            'Soft Exudates': 'SE'
        }

        self.populate_lists()
        if shuffle_list:
            p = [x for x in range(len(self.img_path_list))]
            random.shuffle(p)
            self.img_path_list = [self.img_path_list[pi] for pi in p]
            self.label_path_list = [self.label_path_list[pi] for pi in p]
            self.label_names_text = [self.label_names_text[pi] for pi in p]


        #define data transforms
        self.idrid_transform = IDRID_Transform(config = config)

    def populate_lists(self):
        # print(self.folder_start, self.folder_end, self.label_list)

        for case_no in sorted(os.listdir(os.path.join(self.root_path,'images'))):
            case_idx = int(case_no[case_no.find('_')+1:case_no.find('.')])
            if not((case_idx>=self.folder_start) and (case_idx<self.folder_end)):
                continue
            im_path = os.path.join(self.root_path, 'images',case_no)
            for i in range(len(self.label_list)):
                #need to do this for this dataset
                modified_case_no = case_no[:-4]+'_'+self.acronym[self.label_names[i]]+'.tif'
                label_path = os.path.join(self.root_path, 'labels', self.label_names[i], modified_case_no)
                self.img_path_list.append(im_path)
                self.label_path_list.append(label_path)
                self.label_names_text.append(self.label_names[i])

    def __len__(self):
        assert(len(self.img_path_list)==len(self.label_path_list))
        return len(self.img_path_list)
    
    def __getitem__(self, index):
        img = torch.as_tensor(np.array(Image.open(self.img_path_list[index])))
        try:
            label = torch.Tensor(np.array(Image.open(self.label_path_list[index])))
        except:
            #no label for this image is equivalent to all black label
            label = torch.zeros((self.config['data_transforms']['img_size'], self.config['data_transforms']['img_size']))

        if self.config['data']['volume_channel']==2:
            img = img.permute(2,0,1)
        label = label.unsqueeze(0)

        print("before idrid transform: ", img.shape)
        img, label = self.idrid_transform(img, label, apply_norm=self.apply_norm, is_train = self.is_train)
        print("after idrid transform: ", img.shape)

        
        label_text = self.label_names_text[index]
        label_segmask_no = self.label_list[self.label_names.index(label_text)]

        #idrid has separate masks according to the labels already, so no extra processing needed
        label=label[0]
        label = (label>=0.5)+0

        # print('debug5: ', label.shape, label.any())

        return img, label, label_segmask_no, label_text

class Ultrasound_Dataset(Dataset):
    def __init__(self, config, is_train=False, apply_norm=True, shuffle_list=True, no_text_mode=False):
        super().__init__()
        self.root_path = config['data']['root_path']
        self.img_names = []
        self.img_path_list = []
        self.label_path_list = []
        self.label_list = []
        self.is_train = is_train
        self.label_names = config['data']['label_names']
        self.config = config
        self.apply_norm = apply_norm
        self.no_text_mode = no_text_mode
        self.data_transform = Ultrasound_Transform(config=config)
        self.label_dict = {
            'Liver': [[100,0,100]],
            'Kidney': [[255,255,0]],
            'Pancreas': [[0,0,255]],
            'Vessels': [[255,0,0]],
            'Adrenals': [[0,255,255]],
            'Gall Bladder': [[0,255,0]],
            'Bones': [[255,255,255]],
            'Spleen': [[255,0,255]]
        }
        self.num_classes = len(list(self.label_dict.keys()))
        if self.is_train:
            self.ctlist = ['ct1','ct2','ct3','ct4','ct5','ct6','ct7','ct8','ct9','ct10','ct11','ct12']
        else:
            self.ctlist = ['ct13','ct14','ct15']

        self.populate_lists()
        if shuffle_list:
            p = [x for x in range(len(self.img_path_list))]
            random.shuffle(p)
            self.img_path_list = [self.img_path_list[pi] for pi in p]
            self.img_names = [self.img_names[pi] for pi in p]
            self.label_path_list = [self.label_path_list[pi] for pi in p]
            self.label_list = [self.label_list[pi] for pi in p]

    def populate_lists(self):
        imgs_path = os.path.join(self.root_path, 'images/train')
        labels_path = os.path.join(self.root_path, 'annotations/train')
        for img in os.listdir(imgs_path):
            ct = img[:img.find('-')]
            if ct not in self.ctlist:
                continue
            if self.no_text_mode:
                self.img_names.append(img)
                self.img_path_list.append(os.path.join(imgs_path,img))
                self.label_path_list.append(os.path.join(labels_path, img))
                self.label_list.append('')
            else:
                for label_name in self.label_names:
                    self.img_names.append(img)
                    self.img_path_list.append(os.path.join(imgs_path,img))
                    self.label_path_list.append(os.path.join(labels_path, img))
                    self.label_list.append(label_name)

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        img = torch.as_tensor(np.array(Image.open(self.img_path_list[index]).convert("RGB")))
        try:
            label = (np.array(Image.open(self.label_path_list[index]).convert("RGB")))
        except:
            label = np.zeros(img.shape[0], img.shape[1], 1)

        if self.config['data']['volume_channel']==2:
            img = img.permute(2,0,1)
        
        if self.no_text_mode:
            mask = np.zeros((self.num_classes,img.shape[1], img.shape[2]))
            for i,c in enumerate(list(self.label_dict.keys())):
                temp = np.zeros(label.shape).astype('uint8')[:,:,0]
                selected_color_list = self.label_dict[c]
                for c in selected_color_list:
                    temp = temp | (np.all(np.where(label==c,1,0),axis=2))
                mask[i,:,:] = temp
            mask = torch.Tensor(mask)
            img, mask = self.data_transform(img, mask, is_train=self.is_train, apply_norm=self.apply_norm)
            mask = (mask>=0.5)+0
            label_of_interest = ''
        else:
            temp = np.zeros(label.shape).astype('uint8')[:,:,0]
            selected_color_list = self.label_dict[self.label_list[index]]
            for c in selected_color_list:
                temp = temp | (np.all(np.where(label==c,1,0),axis=2))

            mask = torch.Tensor(temp).unsqueeze(0)
            label_of_interest = self.label_list[index]
            img, mask = self.data_transform(img, mask, is_train=self.is_train, apply_norm=self.apply_norm)
            #convert all grayscale pixels due to resizing back to 0, 1
            mask = (mask>=0.5)+0
            mask = mask[0]

        return img, mask, self.img_path_list[index], label_of_interest    
                


class Cholec_Ins_Dataset(Dataset):
    def __init__(self, config, is_train=False, apply_norm=True, shuffle_list=True, no_text_mode=False) -> None:
        super().__init__()
        self.root_path = config['data']['root_path']
        self.img_names = []
        self.img_path_list = []
        self.label_path_list = []
        self.label_list = []
        self.is_train = is_train
        self.label_names = config['data']['label_names']
        self.config = config
        self.no_text_mode = no_text_mode
        self.apply_norm = apply_norm
        self.data_transform = Cholec_8k_Transform(config=config)
        self.label_dict = {
            'Grasper':31,
            'L Hook Electrocautery':32,
            'Liver':21,
            'Fat':12, 
            'Gall Bladder':22,
            'Abdominal Wall':11,
            'Gastrointestinal Tract':13,
            'Cystic Duct':25,
            'Blood':24,
            'Hepatic Vein':33,
            'Liver Ligament':5,
            'Connective Tissue':23
        }
        self.num_classes = len(list(self.label_dict.keys()))

        if is_train:
            self.folder_list = ['video01','video09','video12','video17','video18','video20','video24','video25', 'video26']
        else:
            self.folder_list = ['video27','video28']
        #populate the above lists
        self.populate_lists()
        if shuffle_list:
            p = [x for x in range(len(self.img_path_list))]
            random.shuffle(p)
            self.img_path_list = [self.img_path_list[pi] for pi in p]
            self.img_names = [self.img_names[pi] for pi in p]
            self.label_path_list = [self.label_path_list[pi] for pi in p]
            self.label_list = [self.label_list[pi] for pi in p]

    def populate_lists(self):
        for folder in (self.folder_list):
            path1 = os.path.join(self.root_path, folder)
            for sub in sorted(os.listdir(path1)):
                path2 = os.path.join(path1, sub)
                for im in sorted(os.listdir(path2)):
                    if 'endo.png' not in im:
                        continue
                    im_path = os.path.join(path2, im)
                    im_name = im[:-4]
                    label_img_path = os.path.join(path2, im_name+'_watershed_mask.png')
                    if self.no_text_mode:
                        self.img_names.append(im_name)
                        self.img_path_list.append(os.path.join(im_path))
                        self.label_path_list.append(os.path.join(label_img_path))
                        self.label_list.append('')
                    else:
                        for label_name in self.label_names:
                            self.img_names.append(im_name)
                            self.img_path_list.append(im_path)
                            self.label_path_list.append(label_img_path)
                            self.label_list.append(label_name)
    
    def __len__(self):
        return len(self.img_path_list)


    def __getitem__(self, index):
        img = torch.as_tensor(np.array(Image.open(self.img_path_list[index]).convert("RGB")))
        if self.config['data']['volume_channel']==2:
            img = img.permute(2,0,1)

        label_of_interest = self.label_list[index]
        gold = np.array(Image.open(self.label_path_list[index]))

        if len(gold.shape)==3:
            gold = gold[:,:,0]
        if gold.max()<2:
            gold = (gold*255).astype(int)


        if self.no_text_mode:
            mask = np.zeros((self.num_classes,img.shape[1], img.shape[2]))
            for i,c in enumerate(list(self.label_dict.keys())):
                mask[i,:,:] = (gold==self.label_dict[c])
            mask = torch.Tensor(mask)
            img, mask = self.data_transform(img, mask, is_train=self.is_train, apply_norm=self.apply_norm)
            mask = (mask>=0.5)+0
            label_of_interest = ''
        else:
            # plt.imshow(gold)
            # plt.show()
            mask = (gold==self.label_dict[label_of_interest])
            
            mask = torch.Tensor(mask+0)
            mask = torch.Tensor(mask).unsqueeze(0)
            

            img, mask = self.data_transform(img, mask, is_train=self.is_train, apply_norm=self.apply_norm)

            # plt.imshow(mask, cmap='gray')
            # plt.show()
            #convert all grayscale pixels due to resizing back to 0, 1
            mask = (mask>=0.5)+0
            mask = mask[0]
            # plt.imshow(mask, cmap='gray')
            # plt.show()
        return img, mask, self.img_path_list[index], label_of_interest

class ChestXDet_Dataset(Dataset):
    def __init__(self, config, start = 0, end = 69565, is_train=False, apply_norm=True, shuffle_list=True, no_text_mode=False) -> None:
        super().__init__()
        self.root_path = config['data']['root_path']
        self.img_names = []
        self.img_path_list = []
        self.label_path_list = []
        self.label_list = []
        self.is_train = is_train
        self.label_names = config['data']['label_names']
        self.config = config
        self.no_text_mode = no_text_mode
        self.apply_norm = apply_norm
        self.start = start
        self.end = end
        self.data_transform = ChestXDet_Transform(config=config)
        self.label_dict = {
            'Effusion': 1, 
            'Nodule': 2, 
            'Cardiomegaly': 3, 
            'Fibrosis': 4, 
            'Consolidation': 5, 
            'Emphysema': 6, 
            'Mass': 7, 
            'Fracture': 8, 
            'Calcification': 9, 
            'Pleural Thickening': 10, 
            'Pneumothorax': 11, 
            'Atelectasis': 12, 
            'Diffuse Nodule': 13
            }
        self.num_classes = len(list(self.label_dict.keys()))

        #populate the above lists
        self.populate_lists()
        if shuffle_list:
            p = [x for x in range(len(self.img_path_list))]
            random.shuffle(p)
            self.img_path_list = [self.img_path_list[pi] for pi in p]
            self.img_names = [self.img_names[pi] for pi in p]
            self.label_path_list = [self.label_path_list[pi] for pi in p]
            self.label_list = [self.label_list[pi] for pi in p]

    def populate_lists(self):
        im_folder_path = os.path.join(self.root_path, 'images')
        mask_folder_path = os.path.join(self.root_path, 'masks')
        for im in os.listdir(im_folder_path):
            if (int(im[:im.find('.')]) >= self.start) and (int(im[:im.find('.')])<=self.end):
                im_path = os.path.join(im_folder_path, im)
                label_img_path = os.path.join(mask_folder_path, im)
                if self.no_text_mode:
                    self.img_names.append(im)
                    self.img_path_list.append(im_path)
                    self.label_path_list.append(label_img_path)
                    self.label_list.append('')
                else:
                    for label_name in self.label_names:
                        self.img_names.append(im)
                        self.img_path_list.append(im_path)
                        self.label_path_list.append(label_img_path)
                        self.label_list.append(label_name)
    
    def __len__(self):
        return len(self.img_path_list)


    def __getitem__(self, index):
        img = torch.as_tensor(np.array(Image.open(self.img_path_list[index]).convert("RGB")))
        if self.config['data']['volume_channel']==2:
            img = img.permute(2,0,1)

        label_of_interest = self.label_list[index]
        gold = np.array(Image.open(self.label_path_list[index]))

        if len(gold.shape)==3:
            gold = gold[:,:,0]

        if self.no_text_mode:
            mask = np.zeros((self.num_classes,img.shape[1], img.shape[2]))
            for i,c in enumerate(list(self.label_dict.keys())):
                mask[i,:,:] = (gold==self.label_dict[c])
            mask = torch.Tensor(mask)
            img, mask = self.data_transform(img, mask, is_train=self.is_train, apply_norm=self.apply_norm)
            mask = (mask>=0.5)+0
            label_of_interest = ''
        else:
            # plt.imshow(gold)
            # plt.show()
            mask = (gold==self.label_dict[label_of_interest])
            
            mask = torch.Tensor(mask+0)
            mask = torch.Tensor(mask).unsqueeze(0)
            

            img, mask = self.data_transform(img, mask, is_train=self.is_train, apply_norm=self.apply_norm)

            # plt.imshow(mask, cmap='gray')
            # plt.show()
            #convert all grayscale pixels due to resizing back to 0, 1
            mask = (mask>=0.5)+0
            mask = mask[0]
            # plt.imshow(mask, cmap='gray')
            # plt.show()
        return img, mask, self.img_path_list[index], label_of_interest


class Endovis_18(Dataset):
    def __init__(self, config, start=0, end=200, is_train=False, shuffle_list = True, apply_norm=True, no_text_mode=False):
        super().__init__()
        self.root_path = config['data']['root_path']
        self.img_names = []
        self.img_path_list = []
        self.label_path_list = []
        self.label_list = []
        self.is_train = is_train
        self.start = start
        self.end = end
        self.label_names = config['data']['label_names']
        self.config = config
        self.no_text_mode = no_text_mode
        self.apply_norm = apply_norm
        if self.is_train:
            self.seqs = ['seq_1', 'seq_2', 'seq_3', 'seq_5', 'seq_6', 'seq_9', 'seq_10', 'seq_11', 'seq_13', 'seq_14', 'seq_15']
        else:
            self.seqs = ['seq_4', 'seq_7', 'seq_12', 'seq_16']

        self.label_dict = {
            'background tissue': [[0,0,0]],
            'surgical instrument': [[0,255,0],[0,255,255],[125,255,12]],
            'kidney parenchyma': [[255,55,0]],
            'covered kidney': [[24,55,125]],
            'thread': [[187,155,25]],
            'clamps': [[0,255,125]],
            'suturing needle': [[255,255,125]],
            'suction instrument': [[123,15,175]],
            'small intestine': [[124,155,5]],
            'ultrasound probe': [[12,255,141]]
        }
        self.num_classes = len(list(self.label_dict.keys()))


        self.populate_lists()
        if shuffle_list:
            p = [x for x in range(len(self.img_path_list))]
            random.shuffle(p)
            self.img_path_list = [self.img_path_list[pi] for pi in p]
            self.img_names = [self.img_names[pi] for pi in p]
            self.label_path_list = [self.label_path_list[pi] for pi in p]
            self.label_list = [self.label_list[pi] for pi in p]

        #define data transform
        self.data_transform = ENDOVIS_18_Transform(config=config)

    def populate_lists(self):
        #generate dataset for instrument 1 4 training
        for dataset_num in os.listdir(self.root_path):
            for seq in os.listdir(os.path.join(self.root_path, dataset_num)):
                if seq not in self.seqs:
                    continue
                lbl_folder_path = os.path.join(self.root_path, dataset_num, seq, 'labels')
                frames_folder_path = os.path.join(self.root_path, dataset_num, seq, 'left_frames')
                for frame_no in os.listdir(frames_folder_path):
                    if 'png' not in frame_no:
                        continue
                    if self.no_text_mode:
                        self.img_names.append(frame_no)
                        self.img_path_list.append(os.path.join(frames_folder_path,frame_no))
                        self.label_path_list.append(os.path.join(lbl_folder_path, frame_no))
                        self.label_list.append('')
                    else:
                        for label_name in self.label_names:
                            lbl_path = os.path.join(lbl_folder_path,frame_no)
                            self.img_names.append(frame_no)
                            self.img_path_list.append(os.path.join(frames_folder_path, frame_no))
                            self.label_list.append(label_name)
                            self.label_path_list.append(lbl_path)

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        img = torch.as_tensor(np.array(Image.open(self.img_path_list[index]).convert("RGB")))
        try:
            label = (np.array(Image.open(self.label_path_list[index]).convert("RGB")))
        except:
            label = np.zeros(img.shape[0], img.shape[1], 1)

        if self.config['data']['volume_channel']==2:
            img = img.permute(2,0,1)
        
        if self.no_text_mode:
            mask = np.zeros((self.num_classes,img.shape[1], img.shape[2]))
            for i,c in enumerate(list(self.label_dict.keys())):
                temp = np.zeros(label.shape).astype('uint8')[:,:,0]
                selected_color_list = self.label_dict[c]
                for c in selected_color_list:
                    temp = temp | (np.all(np.where(label==c,1,0),axis=2))
                mask[i,:,:] = temp
            mask = torch.Tensor(mask)
            img, mask = self.data_transform(img, mask, is_train=self.is_train, apply_norm=self.apply_norm)
            mask = (mask>=0.5)+0
            label_of_interest = ''
        else:
            temp = np.zeros(label.shape).astype('uint8')[:,:,0]
            selected_color_list = self.label_dict[self.label_list[index]]
            for c in selected_color_list:
                temp = temp | (np.all(np.where(label==c,1,0),axis=2))

            mask = torch.Tensor(temp).unsqueeze(0)
            label_of_interest = self.label_list[index]
            img, mask = self.data_transform(img, mask, is_train=self.is_train, apply_norm=self.apply_norm)
            #convert all grayscale pixels due to resizing back to 0, 1
            mask = (mask>=0.5)+0
            mask = mask[0]

        return img, mask, self.img_path_list[index], label_of_interest


class Endovis_Dataset(Dataset):
    def __init__(self, config, start=0, end=200, is_train=False, shuffle_list = True, apply_norm=True, no_text_mode=False):
        super().__init__()
        self.root_path = config['data']['root_path']
        self.img_names = []
        self.img_path_list = []
        self.label_path_list = []
        self.label_list = []
        self.is_train = is_train
        self.start = start
        self.end = end
        self.label_names = config['data']['label_names']
        self.num_classes = len(self.label_names)
        self.config = config
        self.apply_norm = apply_norm
        self.no_text_mode = no_text_mode

        self.populate_lists()
        if shuffle_list:
            p = [x for x in range(len(self.img_path_list))]
            random.shuffle(p)
            self.img_path_list = [self.img_path_list[pi] for pi in p]
            self.img_names = [self.img_names[pi] for pi in p]
            self.label_path_list = [self.label_path_list[pi] for pi in p]
            self.label_list = [self.label_list[pi] for pi in p]

        #define data transform
        self.data_transform = ENDOVIS_Transform(config=config)
    
    def populate_lists(self):
        #generate dataset for instrument 1 4 training
        for dataset_num in os.listdir(self.root_path):
            if 'dataset' not in dataset_num:
                continue
            lbl_folder_path = os.path.join(self.root_path, dataset_num, 'ground_truth')
            frames_folder_path = os.path.join(self.root_path, dataset_num, 'left_frames')
            for frame_no in os.listdir(frames_folder_path):
                if int(frame_no[5:8])>=self.start and int(frame_no[5:8])<self.end:
                    if self.no_text_mode:
                        self.img_names.append(frame_no)
                        self.img_path_list.append(os.path.join(frames_folder_path, frame_no))
                        self.label_path_list.append(lbl_folder_path)
                        self.label_list.append('')
                    else:
                        for label_name in self.label_names:
                            lbl_path = os.path.join(lbl_folder_path, label_name.replace(' ','_')+'_labels',frame_no)
                            
                            #important decision here - include all black labels or not
                            # if not os.path.exists(lbl_path):
                            #     continue
                            self.img_names.append(frame_no)
                            self.img_path_list.append(os.path.join(frames_folder_path, frame_no))
                            self.label_list.append(label_name)
                            self.label_path_list.append(lbl_path)

    def __len__(self):
        return len(self.img_path_list)
    
    def __getitem__(self, index):
        img = torch.as_tensor(np.array(Image.open(self.img_path_list[index]).convert("RGB")))
        if self.config['data']['volume_channel']==2:
            img = img.permute(2,0,1)

        if self.no_text_mode:
            label = torch.zeros((self.num_classes,img.shape[1],img.shape[2]))
            for i,label_name in enumerate(self.label_names):
                try:
                    lbl_path = os.path.join(self.label_path_list[index],label_name.replace(' ','_')+'_labels',self.img_names[index])
                    # print("lbl path: ", lbl_path)
                    label_part = torch.Tensor(np.array(Image.open(lbl_path)))
                except:
                    label_part = torch.zeros(img.shape[1], img.shape[2])
                label[i,:,:] = label_part
            label = (label>0)+0
            img, label = self.data_transform(img, label, is_train=self.is_train, apply_norm=self.apply_norm)
            label = (label>=0.5)+0
            label_of_interest = ''
            # print("img shape: ",img.shape)
            # print("label shape: ", label.shape)
            
        else:
            try:
                label = torch.Tensor(np.array(Image.open(self.label_path_list[index])))
            except:
                label = torch.zeros(img.shape[0], img.shape[1])

            
            label = label.unsqueeze(0)
            label = (label>0)+0
            label_of_interest = self.label_list[index]
            img, label = self.data_transform(img, label, is_train=self.is_train, apply_norm=self.apply_norm)

            #convert all grayscale pixels due to resizing back to 0, 1
            img, label = self.data_transform(img, label, is_train=self.is_train, apply_norm=self.apply_norm)
            label = (label>=0.5)+0
            label = label[0]


        return img, label, self.img_path_list[index], label_of_interest

    def __len__(self):
        return len(self.img_path_list)

    

class KvasirSeg_Dataset(Dataset):
    def __init__(self, config, is_train=False, shuffle_list = True, apply_norm=True, no_text_mode=False):
        super().__init__()
        self.root_path = config['data']['root_path']
        self.img_names = []
        self.img_path_list = []
        self.label_path_list = []
        self.label_list = []
        self.is_train = is_train
        self.label_names = config['data']['label_names']
        self.num_classes = len(self.label_names)
        self.config = config
        self.apply_norm = apply_norm
        self.no_text_mode = no_text_mode

        self.populate_lists()
        if shuffle_list:
            p = [x for x in range(len(self.img_path_list))]
            random.shuffle(p)
            self.img_path_list = [self.img_path_list[pi] for pi in p]
            self.img_names = [self.img_names[pi] for pi in p]
            self.label_path_list = [self.label_path_list[pi] for pi in p]
            self.label_list = [self.label_list[pi] for pi in p]

        #define data transform
        self.data_transform = kvasirSeg_Transform(config=config)
    
    def __len__(self):
        return len(self.img_path_list)
    
    def populate_lists(self):

        if self.is_train:
            imgs_path = os.path.join(self.root_path, "train/images")
            masks_path = os.path.join(self.root_path, "train/masks")
        else:
            imgs_path = os.path.join(self.root_path, "val/images")
            masks_path = os.path.join(self.root_path, "val/masks")
        
        for i in os.listdir(imgs_path):
            if self.no_text_mode:
                self.img_names.append(i)
                self.img_path_list.append(os.path.join(imgs_path,i))
                self.label_path_list.append(os.path.join(masks_path, i))
                self.label_list.append('')
            else:
                for label_name in self.label_names:
                    self.img_names.append(i)
                    self.img_path_list.append(os.path.join(imgs_path,i))
                    self.label_path_list.append(os.path.join(masks_path, i))
                    self.label_list.append(label_name)

    def __getitem__(self, index):
        img = torch.as_tensor(np.array(Image.open(self.img_path_list[index]).convert("RGB")))
        if self.config['data']['volume_channel']==2:
            img = img.permute(2,0,1)
            
        try:
            label = torch.Tensor(np.array(Image.open(self.label_path_list[index])))[:,:,0]
        except:
            label = torch.zeros(img.shape[1], img.shape[2])

        
        label = label.unsqueeze(0)
        label = (label>0)+0
        label_of_interest = self.label_list[index]
        img, label = self.data_transform(img, label, is_train=self.is_train, apply_norm=self.apply_norm)

        #convert all grayscale pixels due to resizing back to 0, 1
        img, label = self.data_transform(img, label, is_train=self.is_train, apply_norm=self.apply_norm)
        label = (label>=0.5)+0
        label = label[0]


        return img, label, self.img_path_list[index], label_of_interest

def get_data(config, tr_folder_start, tr_folder_end, val_folder_start, val_folder_end, use_norm=True, no_text_mode=False):
    dataset_dict = {}
    dataloader_dict = {}
    dataset_sizes = {}
    #generate label_dict
    label_dict = {}
    for i,ln in enumerate(config['data']['label_names']):
        label_dict[ln] = i

    if config['data']['name']=='IDRID':
        for x in ['train','val']:
            if x=='train':
                dataset_dict[x] = IDRID_Dataset(config, folder_start=0, folder_end=40, shuffle_list=True, is_train=True, apply_norm=use_norm)
            if x=='val':
                dataset_dict[x] = IDRID_Dataset(config, folder_start=40, folder_end=60, shuffle_list=False, apply_norm=use_norm)
            dataset_sizes[x] = len(dataset_dict[x])

    elif config['data']['name']=='ENDOVIS':
        for x in ['train','val']:
            if x=='train':
                dataset_dict[x] = Endovis_Dataset(config, start=0, end=180, shuffle_list=True, is_train=True, apply_norm=use_norm, no_text_mode=no_text_mode)
            if x=='val':
                dataset_dict[x] = Endovis_Dataset(config, start=180, end=330, shuffle_list=False, apply_norm=use_norm, no_text_mode=no_text_mode)
            dataset_sizes[x] = len(dataset_dict[x])

    elif config['data']['name']=='ENDOVIS 18':
        for x in ['train','val']:
            if x=='train':
                dataset_dict[x] = Endovis_18(config, start=0, end=18000, shuffle_list=True, is_train=True, apply_norm=use_norm, no_text_mode=no_text_mode)
            if x=='val':
                dataset_dict[x] = Endovis_18(config, start=0, end=33000, shuffle_list=False, apply_norm=use_norm, is_train=False, no_text_mode=no_text_mode)
            dataset_sizes[x] = len(dataset_dict[x])

    elif config['data']['name']=='CHESTXDET':
        for x in ['train','val']:
            if x=='train':
                dataset_dict[x] = ChestXDet_Dataset(config, start=0, end=69565, shuffle_list=True, is_train=True, apply_norm=use_norm, no_text_mode=no_text_mode)
            if x=='val':
                dataset_dict[x] = ChestXDet_Dataset(config, start=69566, end=83000, shuffle_list=False, apply_norm=use_norm, is_train=False, no_text_mode=no_text_mode)
            dataset_sizes[x] = len(dataset_dict[x])

    elif config['data']['name']=='CHOLEC 8K':
        for x in ['train','val']:
            if x=='train':
                dataset_dict[x] = Cholec_Ins_Dataset(config, shuffle_list=True, is_train=True, apply_norm=use_norm, no_text_mode=no_text_mode)
            if x=='val':
                dataset_dict[x] = Cholec_Ins_Dataset(config, shuffle_list=False, apply_norm=use_norm, is_train=False, no_text_mode=no_text_mode)
            dataset_sizes[x] = len(dataset_dict[x])
    
    elif config['data']['name']=='ULTRASOUND':
        for x in ['train','val']:
            if x=='train':
                dataset_dict[x] = Ultrasound_Dataset(config, shuffle_list=True, is_train=True, apply_norm=use_norm, no_text_mode=no_text_mode)
            if x=='val':
                dataset_dict[x] = Ultrasound_Dataset(config, shuffle_list=False, apply_norm=use_norm, is_train=False, no_text_mode=no_text_mode)
            dataset_sizes[x] = len(dataset_dict[x])

    elif config['data']['name']=='KVASIRSEG':
        for x in ['train','val']:
            if x=='train':
                dataset_dict[x] = KvasirSeg_Dataset(config, shuffle_list=True, is_train=True, apply_norm=use_norm, no_text_mode=no_text_mode)
            if x=='val':
                dataset_dict[x] = KvasirSeg_Dataset(config, shuffle_list=False, apply_norm=use_norm, is_train=False, no_text_mode=no_text_mode)
            dataset_sizes[x] = len(dataset_dict[x])

    else:
        for x in ['train','val']:
            if x=='train':
                dataset_dict[x] = Generic_Dataset_3d(config, is_train=True, folder_start=tr_folder_start, folder_end=tr_folder_end)
            elif x=='val':
                dataset_dict[x] = Generic_Dataset_3d(config, is_train=False, folder_start=val_folder_start, folder_end=val_folder_end)

            dataset_sizes[x] = len(dataset_dict[x])
    return dataset_dict, dataset_sizes, label_dict