import argparse
import yaml
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from data_utils import *
from model import *
from test import *
from train import *

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_config', default='config_tmp.yml',
                        help='data config file path')

    parser.add_argument('--model_config', default='model_baseline.yml',
                        help='model config file path')

    parser.add_argument('--pretrained_path', default=None,
                        help='pretrained model path')

    parser.add_argument('--save_path', default='checkpoints/temp.pth',
                        help='pretrained model path')
    parser.add_argument('--training_strategy', default='biastuning', help='how to train the model')

    args = parser.parse_args()

    return args

def main_datautils(config):
    print(config)
    dataset_dict, dataset_sizes, label_dict = get_data(config, tr_folder_start=0, tr_folder_end=78, val_folder_start=78, val_folder_end=104)
    print(len(dataset_dict['train']))
    temp = (dataset_dict['train'][0])
    print(temp[0].shape)
    print(temp[1].shape)
    plt.imshow(temp[1], cmap='gray')
    plt.show()
    print(temp[-1])

def main_model(config):
    print(config)
    label_dict = {
        'liver':0,
        'tumor':1
    }
    model = Prompt_Adapted_SAM(config, label_dict)
    for name, p in model.named_parameters():
        print(name)
    return

def main_test(data_config, model_config, pretrained_path):
    test_start = 104
    test_end = 131
    test(data_config, model_config, pretrained_path, test_start, test_end, device='cuda:0')

def main_train(data_config, model_config, pretrained_path, save_path, training_strategy='biastuning', device='cuda:0'):
    #load data
    if data_config['data']['name']=='LITS':
        dataset_dict, dataset_sizes, label_dict = get_data(data_config, tr_folder_start=0, tr_folder_end=78, val_folder_start=78, val_folder_end=104)
    elif data_config['data']['name']=='IDRID':
        dataset_dict, dataset_sizes, label_dict = get_data(data_config, tr_folder_start=0, tr_folder_end=40, val_folder_start=40, val_folder_end=104)
        dataloader_dict = {}
        for x in ['train','val']:
            dataloader_dict[x] = torch.utils.data.DataLoader(dataset_dict[x], batch_size=model_config['training']['batch_size'], shuffle=True, num_workers=4)

    #load model
    if model_config['arch']=='Prompt Adapted SAM':
        model = Prompt_Adapted_SAM(model_config, label_dict, device)

    #load model weights
    if pretrained_path is not None:
        model.load_state_dict(torch.load(pretrained_path))

    #freeze correct weights
    for p in model.parameters():
        p.requires_grad=False
    if training_strategy=='biastuning':
        for name, p in model.named_parameters():
            if 'bias' in name and 'clip' not in name:
                p.requires_grad = True
    elif training_strategy=='prompt_tuning':
        for name,p in model.named_parameters():
            if 'prompt' in name:
                p.requires_grad = True
            if 'decoder' in name:
                p.requires_grad = True
            if 'Text_Embedding_Affine' in name:
                p.requires_grad = True
    elif training_strategy=='fdn':
        for name,p in model.named_parameters():
            if 'FDN' in name:
                p.requires_grad = True

    #training parameters
    print('number of trainable parameters: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    training_params = model_config['training']
    if training_params['optimizer'] == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=float(training_params['lr']), weight_decay=float(training_params['weight_decay']))
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=training_params['schedule_step'], gamma=training_params['schedule_step_factor'])
    
    if training_params['loss']=='dice':
        criterion = [dice_loss]
    elif training_params['loss']=='dice+CE':
        criterion = [dice_loss, nn.BCELoss()]
    else:
        criterion = [nn.BCELoss()]
    
    #train the model
    if data_config['data']['name']=='LITS':
        model = train(model, dataset_dict['train'], dataset_dict['val'], criterion, optimizer, save_path, num_epochs=training_params['num_epochs'], bs=training_params['batch_size'], device=device)
    elif data_config['data']['name']=='IDRID':
        model = train_dl(model, dataloader_dict, dataset_sizes, criterion, optimizer, exp_lr_scheduler, save_path, num_epochs=training_params['num_epochs'], bs=training_params['batch_size'], device=device)



if __name__ == '__main__':
    args = parse_args()
    with open(args.data_config, 'r') as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.model_config, 'r') as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
    
    # #for checking data_utils
    # main_datautils(data_config)

    # #for checking model
    # main_model(config=model_config)

    # #for testing on the test dataset
    # main_test(data_config, model_config, args.pretrained_path)

    # for training the model
    main_train(data_config, model_config, args.pretrained_path, args.save_path, args.training_strategy, device='cuda:0')
