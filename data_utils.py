import random
import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
from PIL import Image
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


class IDRID_Dataset(Dataset):
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
        self.idrid_transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.PILToTensor(),
            transforms.RandomAdjustSharpness(2,0.5),
            transforms.RandomAdjustSharpness(0.5,0.5),
            # transforms.RandomEqualize(0.5),
            # transforms.RandomRotation(15)
        ])
        self.transform = Slice_Transforms(config=config)


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
        img = Image.open(self.img_path_list[index])
        # img = torch.Tensor(np.array(Image.open(self.img_path_list[index])))
        # if self.config['data']['volume_channel']==2:
        #     img = img.permute(2,0,1)
        if self.is_train:
            img = self.idrid_transform(img)
        # print("debug0: ",img.shape)
        else:
            img = torch.Tensor(np.array(img))
            if self.config['data']['volume_channel']==2:
                img = img.permute(2,0,1)
        img = img.unsqueeze(0)
        # print("debug1: ",img.shape)

        img = self.transform(img)
        img = img[0]
        

        try:
            label = torch.Tensor(np.array(Image.open(self.label_path_list[index])))
        except:
            #no label for this image is equivalent to all black label
            label = torch.zeros((self.config['data_transforms']['img_size'], self.config['data_transforms']['img_size']))
        label_text = self.label_names_text[index]
        label_segmask_no = self.label_list[self.label_names.index(label_text)]

        #convert general mask into prompted segmentation mask per according to label name
        # print('debug3: ', label.shape)
        label = label.unsqueeze(0).unsqueeze(0)
        gold = self.transform(label, is_mask=True)
        # print('debug4: ', gold.shape)
        gold=gold[0,0]
        gold = (gold>=0.5)+0

        # print('debug5: ', gold.shape, gold.any())

        return img, gold, label_segmask_no, label_text


def get_data(config, tr_folder_start, tr_folder_end, val_folder_start, val_folder_end):
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
                dataset_dict[x] = IDRID_Dataset(config, folder_start=0, folder_end=40, shuffle_list=True, is_train=True)
            if x=='val':
                dataset_dict[x] = IDRID_Dataset(config, folder_start=40, folder_end=60, shuffle_list=False)
            dataset_sizes[x] = len(dataset_dict[x])

    else:
        for x in ['train','val']:
            if x=='train':
                dataset_dict[x] = Generic_Dataset_3d(config, is_train=True, folder_start=tr_folder_start, folder_end=tr_folder_end)
            elif x=='val':
                dataset_dict[x] = Generic_Dataset_3d(config, is_train=False, folder_start=val_folder_start, folder_end=val_folder_end)

            dataset_sizes[x] = len(dataset_dict[x])
    return dataset_dict, dataset_sizes, label_dict