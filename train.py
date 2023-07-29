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
                    seg_loss=0
                    for c in criterion:
                        seg_loss += c(outputs, labels.float())
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

def train_dl(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, sav_path='./checkpoints/temp.pth', num_epochs=25, bs=32, device='cuda:0'):
    model = model.to(device)
    best_dice = 0
    best_loss=10000

    print("Training parameters: \n----------")
    print("batch size: ", bs)
    print("num epochs: ", num_epochs)
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_intersection = 0
            running_union = 0
            running_corrects = 0
            running_dice = 0
            intermediate_count = 0
            count = 0
            preds_all = []
            gold = []

            # Iterate over data.
            for inputs, labels,text_idxs, text in dataloaders[phase]:
                count+=1
                intermediate_count += inputs.shape[0]

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs, text)
                    # print(outputs)
                    # print(outputs.shape)
                    # print(outputs)
                    loss=0
                    seg_loss = 0
                    for c in criterion:
                        try:
                            seg_loss += c(outputs, text, labels.float())
                        except:
                            seg_loss += c(outputs, labels.float())
                    loss += seg_loss
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                with torch.no_grad():
                    preds = (outputs>=0.5)
                    # preds_all.append(preds.cpu())
                    # gold.append(labels.cpu())
                    # epoch_dice = dice_coef(preds,labels)
                    # if count%100==0:
                    #     print('iteration dice: ', epoch_dice)


                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    ri, ru = running_stats(labels,preds)
                    running_dice += dice_collated(ri,ru)
                    # if count%5==0:
                    #     print(count)
                    #     print(running_loss, intermediate_count)
                    #     print(running_loss/intermediate_count)
            
            if phase == 'train':
                scheduler.step()
            print("all 0 sanity check for preds: ", preds.any())
            print("all 1 sanity check for preds: ", not preds.all())
            epoch_loss = running_loss / ((dataset_sizes[phase]))
            epoch_dice = running_dice / ((dataset_sizes[phase]))
            # epoch_dice = dice_coef(torch.cat(preds_all,axis=0),torch.cat(gold,axis=0))
            print(f'{phase} Loss: {epoch_loss:.4f} Dice: {epoch_dice:.4f}')            

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_dice = epoch_dice
                torch.save(model.state_dict(),sav_path)
            
            elif phase == 'val' and np.isnan(epoch_loss):
                print("nan loss but saving model")
                torch.save(model.state_dict(),sav_path)


    print(f'Best val loss: {best_loss:4f}, best val accuracy: {best_dice:2f}')

    # load best model weights
    # model.load_state_dict(best_model_wts)
    return model