import torch
import yaml
import sys
import copy
import os
sys.path.append("/home/ubuntu/Desktop/Domain_Adaptation_Project/repos/biastuning/")

from data_utils import *
from model import *
from utils import *

label_names = ['Left Prograsp Forceps', 'Maryland Bipolar Forceps', 'Right Prograsp Forceps', 'Left Large Needle Driver', 'Right Large Needle Driver', 'Left Grasping Retractor', 'Right Grasping Retractor', 'Vessel Sealer', 'Monopolar Curved Scissors']
visualize_li = [[1,0,0],[0,1,0],[1,0,0], [0,0,1], [0,0,1]]
label_dict = {}
visualize_dict = {}
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

    parser.add_argument('--labels_of_interest', default='Left Prograsp Forceps,Maryland Bipolar Forceps,Right Prograsp Forceps,Left Large Needle Driver,Right Large Needle Driver', help='labels of interest')

    parser.add_argument('--codes', default='1,2,1,3,3', help='numeric label to save per instrument')

    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    with open(args.data_config, 'r') as f:
        data_config = yaml.load(f, Loader=yaml.FullLoader)
    with open(args.model_config, 'r') as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
    labels_of_interest = args.labels_of_interest.split(',')
    codes = args.codes.split(',')
    codes = [int(c) for c in codes]


    #make folder to save visualizations
    os.makedirs(os.path.join(args.save_path,"preds"),exist_ok=True)
    os.makedirs(os.path.join(args.save_path,"rescaled_preds"),exist_ok=True)
    if args.gt_path:
        os.makedirs(os.path.join(args.save_path,"rescaled_gt"),exist_ok=True)

    #load model
    model = Prompt_Adapted_SAM(config=model_config, label_text_dict=label_dict, device=args.device)
    model.load_state_dict(torch.load(args.pretrained_path, map_location=args.device))
    model = model.to(args.device)
    model = model.eval()

    #load data transform
    data_transform = ENDOVIS_Transform(config=data_config)

    #dice
    dices = []

    #load data
    for img_name in sorted(os.listdir(args.data_folder)):
        img_path = (os.path.join(args.data_folder,img_name))
        if args.gt_path:
            label_name = labels_of_interest[0].replace(' ','_')+'_labels'
            #for test data, the labels are arranged differently so uncomment the line below 
            gt_path = (os.path.join(args.gt_path,img_name))
            # gt_path = (os.path.join(args.gt_path,label_name,img_name))

        # print(img_path)
        img = torch.as_tensor(np.array(Image.open(img_path).convert("RGB")))
        img = img.permute(2,0,1)
        C,H,W = img.shape
        #make a dummy mask of shape 1XHXW
        if args.gt_path:
            label = torch.as_tensor(np.array(Image.open(gt_path))).unsqueeze(0)
        
            #for test data, the labels are arranged differently so uncomment th line below
            label = (label==codes[0])+0

            label = (label>0)+0
        else:
            label = torch.zeros((1,H,W))
        img, label = data_transform(img, label, is_train=False, apply_norm=True)
        label = (label>0.5)+0

        #get image embeddings
        img = img.unsqueeze(0).to(args.device)  #1XCXHXW
        img_embeds = model.get_image_embeddings(img)

        # generate masks for all labels of interest
        img_embeds_repeated = img_embeds.repeat(len(labels_of_interest),1,1,1)
        x_text = [t for t in labels_of_interest]
        masks = model.get_masks_for_multiple_labels(img_embeds_repeated, x_text).cpu()
        argmax_masks = torch.argmax(masks, dim=0)
        final_mask = torch.zeros(masks[0].shape)
        final_mask_rescaled = torch.zeros(masks[0].shape).unsqueeze(-1).repeat(1,1,3)
        #save masks
        for i in range(final_mask.shape[0]):
            for j in range(final_mask.shape[1]):
                final_mask[i,j] = codes[argmax_masks[i,j]] if masks[argmax_masks[i,j],i,j]>=0.5 else 0
                # final_mask_rescaled[i,j] = torch.Tensor(visualize_dict[(labels_of_interest[argmax_masks[i,j]])] if masks[argmax_masks[i,j],i,j]>=0.5 else [0,0,0])

        # save_im = Image.fromarray(final_mask.numpy())
        # save_im.save(os.path.join(args.save_path,'preds', img_name))

        # plt.imshow(final_mask_rescaled,cmap='gray')
        # plt.savefig(os.path.join(args.save_path,'rescaled_preds', img_name))
        # plt.close()

        # print("label shape: ", label.shape)
        # plt.imshow(label[0], cmap='gray')
        # plt.show()

        plt.imshow((masks[0]>0.5), cmap='gray')
        plt.savefig(os.path.join(args.save_path,'rescaled_preds', img_name))
        plt.close()

        if args.gt_path:
            plt.imshow((label[0]), cmap='gray')
            plt.savefig(os.path.join(args.save_path,'rescaled_gt', img_name))
            plt.close()

        # print("dice: ",dice_coef(label, (masks>0.5)+0))
        dices.append(dice_coef(label, (masks>0.5)+0))
        # break
    print(torch.mean(torch.Tensor(dices)))

if __name__ == '__main__':
    main()


        


