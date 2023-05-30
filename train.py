import torch

import sys
import copy
import os

from data_utils import *
from model import *
from utils import *


def train(model, tr_dataset, val_dataset, criterion, optimizer, sav_path='./checkpoints/temp.pth', num_epochs=25, bs=32, device='cuda:0'):
    model = model.to(device)
    best_loss = 100000.0
    best_dice = 0
    print("Training parameters: \n----------")
    print("batch size: ", bs)
    print("num epochs: ", num_epochs)
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        bs_count=0
        inputs_li, labels_li, text_ids_li, text_li = [], [], [], []
        running_loss = 0
        running_dice = 0
        count = 0
        #run training
        # print("eere: ",len(tr_dataset))
        for i in range(len(tr_dataset)):
            inputs, labels,_, text = tr_dataset[i]
            inputs_li.append(inputs)
            labels_li.append(labels)
            text_li = text_li + [text]*(inputs.shape[0])
            bs_count += 1
            if (bs_count%bs==0) or (i==len(tr_dataset)-1):
                #start training
                bs_count=0
                inputs = torch.cat(inputs_li,dim=0)
                labels = torch.cat(labels_li, dim=0)
                inputs = inputs.to(device)
                labels = labels.to(device)
                with torch.set_grad_enabled(True):
                    optimizer.zero_grad()
                    outputs = model(inputs, text_li)
                    seg_loss = criterion(outputs, labels.float())
                    seg_loss.backward()
                    optimizer.step()
                    running_loss += seg_loss.cpu()
                
                preds = (outputs>=0.5)
                ri, ru = running_stats(labels,preds)
                running_dice += dice_collated(ri,ru)
                count += ri.shape[0]
                
                inputs_li = []
                labels_li = []
                text_li = []
        epoch_dice = running_dice / count
        
        print("Training loss: ", running_loss/(1+(len(tr_dataset)//bs)))
        print("Training dice: ", epoch_dice)

        #do val if epoch is a multiple of 5
        if epoch%5==0:
            running_dice = 0
            count=0
            for i in range(len(val_dataset)):
                inputs, labels,_, text = val_dataset[i]
                inputs_li.append(inputs)
                labels_li.append(labels)
                text_li = text_li + [text]*(inputs.shape[0])
                bs_count += 1
                if bs_count%bs==0:
                    #start training
                    bs_count=0
                    inputs = torch.cat(inputs_li,dim=0)
                    labels = torch.cat(labels_li, dim=0)
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    with torch.set_grad_enabled(False):
                        outputs = model(inputs, text_li)
                        preds = (outputs>=0.5)
                        ri, ru = running_stats(labels,preds)
                        running_dice += dice_collated(ri,ru)
                        count += ri.shape[0]

                    inputs_li = []
                    labels_li = []
                    text_li = []
            # epoch_dice = running_dice / (len(val_dataset))
            epoch_dice = running_dice / count

            print(f'Val Dice: {epoch_dice:.4f}')            

            # deep copy the model
            if epoch_dice > best_dice:
                # best_loss = epoch_loss
                best_dice = epoch_dice
                torch.save(model.state_dict(),sav_path)

    return model
