import torch
import yaml
import sys
import copy
import os
sys.path.append("/home/ubuntu/Desktop/Domain_Adaptation_Project/repos/biastuning/")

from data_utils import *
from model import *
from utils import *
from baselines import UNet, UNext, medt_net
from vit_seg_modeling import VisionTransformer
from vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from axialnet import MedT

label_names = ['Liver', 'Kidney', 'Pancreas', 'Vessels', 'Adrenals', 'Gall Bladder', 'Bones', 'Spleen']
# visualize_li = [[1,0,0],[0,1,0],[1,0,0], [0,0,1], [0,0,1]]
label_dict = {}
# visualize_dict = {}
for i,ln in enumerate(label_names):
        label_dict[ln] = i
        # visualize_dict[ln] = visualize_li[i]

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_folder', default='config_tmp.yml',
                        help='data folder file path')

    parser.add_argument('--data_config', default='config_tmp.yml',
                        help='data config file path')

    parser.add_argument('--model_config', default='model_baseline.yml',
                        help='model config file path')

    parser.add_argument('--pretrained_path', default=None,
                        help='pretrained model path')

    parser.add_argument('--save_path', default='checkpoints/temp.pth',
                        help='pretrained model path')

    parser.add_argument('--gt_path', default='',
                        help='ground truth path')

    parser.add_argument('--device', default='cuda:0', help='device to train on')

    parser.add_argument('--codes', default='1,2,1,3,3', help='numeric label to save per instrument')

    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    with open(args.data_config, 'r') as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.model_config, 'r') as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
    codes = args.codes.split(',')
    codes = [int(c) for c in codes]

    label_dict = {
            'Liver': [[100,0,100]],
            'Kidney': [[255,255,0]],
            'Pancreas': [[0,0,255]],
            'Vessels': [[255,0,0]],
            'Adrenals': [[0,255,255]],
            'Gall Bladder': [[0,255,0]],
            'Bones': [[255,255,255]],
            'Spleen': [[255,0,255]]
        }


    #make folder to save visualizations
    os.makedirs(os.path.join(args.save_path,"preds"),exist_ok=True)
    os.makedirs(os.path.join(args.save_path,"rescaled_preds"),exist_ok=True)
    if args.gt_path:
        os.makedirs(os.path.join(args.save_path,"rescaled_gt"),exist_ok=True)


    #load model
    #change the img size in model config according to data config
    in_channels = model_config['in_channels']
    out_channels = model_config['num_classes']
    img_size = model_config['img_size']
    if model_config['arch']=='Prompt Adapted SAM':
        model = Prompt_Adapted_SAM(model_config, label_dict, args.device, training_strategy='biastuning')
    elif model_config['arch']=='UNet':
        model = UNet(in_channels=in_channels, out_channels=out_channels)
    elif model_config['arch']=='UNext':
        model = UNext(num_classes=out_channels, input_channels=in_channels, img_size=img_size)
    elif model_config['arch']=='MedT':
        #TODO
        model = MedT(img_size=img_size, num_classes=out_channels)
    elif model_config['arch']=='TransUNet':
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        config_vit.n_classes = out_channels
        config_vit.n_skip = 3
        # if args.vit_name.find('R50') != -1:
        #     config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
        model = VisionTransformer(config_vit, img_size=img_size, num_classes=config_vit.n_classes)

    model.load_state_dict(torch.load(args.pretrained_path, map_location=args.device))
    model = model.to(args.device)
    model = model.eval()

    #load data transform
    data_transform = Ultrasound_Transform(config=data_config)

    #dice
    dices = []
    ious=[]

    #load data
    for i,img_name in enumerate(sorted(os.listdir(args.data_folder))):
        # if i%5!=0:
        #     continue
        img_path = (os.path.join(args.data_folder,img_name))
        if args.gt_path:
            gt_path = (os.path.join(args.gt_path,img_name))
            if not os.path.exists(gt_path):
                gt_path = (os.path.join(args.gt_path,img_name[:-4]+'.png'))
                if not os.path.exists(gt_path):
                    continue

        # print(img_path)
        img = torch.as_tensor(np.array(Image.open(img_path).convert("RGB")))
        img = img.permute(2,0,1)
        C,H,W = img.shape
        #make a dummy mask of shape 1XHXW
        label = np.array(Image.open(gt_path).convert("RGB"))

        if args.gt_path:

            mask = np.zeros((len(label_dict),img.shape[1], img.shape[2]))
            for i,c in enumerate(list(label_dict.keys())):
                temp = np.zeros(label.shape).astype('uint8')[:,:,0]
                selected_color_list = label_dict[c]
                for c in selected_color_list:
                    temp = temp | (np.all(np.where(label==c,1,0),axis=2))
                mask[i,:,:] = temp
            mask = torch.Tensor(mask)

        else:
            mask = torch.zeros((len(label_dict),H,W))
        img, mask = data_transform(img, mask, is_train=False, apply_norm=True)
        mask = (mask>=0.5)+0

        img = img.unsqueeze(0).to(args.device)  #1XCXHXW
        masks = model(img,'')
        # print("masks shape: ",masks.shape)

        argmax_masks = torch.argmax(masks, dim=1).cpu().numpy()
        # print("argmax masks shape: ",argmax_masks.shape)

        classwise_dices = []
        classwise_ious = []
        for j,c1 in enumerate(label_dict):
            res = np.where(argmax_masks==j,1,0)
            # print("res shape: ",res.shape)
            plt.imshow(res[0], cmap='gray')
            save_dir = os.path.join(args.save_path, c1, 'rescaled_preds')
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(args.save_path, c1, 'rescaled_preds', img_name))
            plt.close()

            if args.gt_path:
                plt.imshow((mask[j]), cmap='gray')
                save_dir = os.path.join(args.save_path, c1, 'rescaled_gt')
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(os.path.join(args.save_path, c1, 'rescaled_gt', img_name))
                plt.close()

                classwise_dices.append(dice_coef(mask[j], torch.Tensor(res[0])))
                classwise_ious.append(iou_coef(mask[j], torch.Tensor(res[0])))

        # break
        dices.append(classwise_dices)
        ious.append(classwise_ious)
        # print("classwise_dices: ", classwise_dices)
        # print("classwise ious: ", classwise_ious)

    print(torch.mean(torch.Tensor(dices),dim=0))
    print(torch.mean(torch.Tensor(ious),dim=0))

if __name__ == '__main__':
    main()


        


