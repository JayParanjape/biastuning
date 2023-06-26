import numpy as np

PATCH_SIZE = 256  # Size of the patches
OVERLAP = 32  # Amount of overlap between patches

def split_image_into_patches(image):
    height, width, _ = image.shape
    patches = []
    
    for y in range(0, height-PATCH_SIZE+1, PATCH_SIZE-OVERLAP):
        for x in range(0, width-PATCH_SIZE+1, PATCH_SIZE-OVERLAP):
            patch = (y,x,image[y:y+PATCH_SIZE, x:x+PATCH_SIZE])
            patches.append(patch)
    
    return patches

def stitch_patches_to_image(patches, image_shape):
    stitched_image = np.zeros(image_shape)
    overlap_mask = np.zeros(image_shape[:2])+1e-10
    
    for patch in patches:
        y, x, p = patch
        try:
            # Add the patch to the stitched image
            stitched_image[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += p
            overlap_mask[y:y+PATCH_SIZE, x:x+PATCH_SIZE] += 1
        except:
            print(p.shape)
            print(y,x)
            print(image_shape)
            1/0
        
    # Normalize the stitched image by dividing with the overlap count
    stitched_image = ((stitched_image/overlap_mask)>0.5)+0
    
    return stitched_image.astype(np.uint8)

import torch
import yaml
import sys
import copy
import os
sys.path.append("/home/ubuntu/Desktop/Domain_Adaptation_Project/repos/biastuning/")

from data_utils import *
from model import *
from utils import *

label_names = ['Left Prograsp Forceps', 'Maryland Bipolar Forceps', 'Right Prograsp Forceps', 'Left Large Needle Driver', 'Right Large Needle Driver']
visualize_li = [[1,0,0],[0,1,0],[1,0,0], [0,0,1], [0,0,1]]
label_dict = {}
visualize_dict = {}
for i,ln in enumerate(label_names):
        label_dict[ln] = i
        visualize_dict[ln] = visualize_li[i]

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

    #load model
    model = Prompt_Adapted_SAM(config=model_config, label_text_dict=label_dict, device=args.device)
    model.load_state_dict(torch.load(args.pretrained_path, map_location=args.device))
    model = model.eval()
    model = model.to(args.device)

    #load data transform
    data_transform = ENDOVIS_Transform(config=data_config)

    #load data
    for img_name in sorted(os.listdir(args.data_folder)):
        img_path = (os.path.join(args.data_folder,img_name))
        # print(img_path)
        original_img = torch.as_tensor(np.array(Image.open(img_path).convert("RGB")))
        patches = split_image_into_patches(original_img)
        patch_masks = []

        for y,x,p in patches:
            img = p.permute(2,0,1)
            #make a dummy mask of shape 1XHXW
            label = torch.zeros(img.shape)[0].unsqueeze(0)
            img, _ = data_transform(img, label, is_train=False, apply_norm=True, crop=False, resize=False)

            #get image embeddings
            img = img.unsqueeze(0).to(args.device)  #1XCXHXW
            img_embeds = model.get_image_embeddings(img)

            # generate masks for all labels of interest
            img_embeds_repeated = img_embeds.repeat(len(labels_of_interest),1,1,1)
            x_text = [t for t in labels_of_interest]
            masks = model.get_masks_for_multiple_labels(img_embeds_repeated, x_text).cpu()

            #for now, only handle one class at a time
            masks, max_idxs = torch.max(masks,dim=0)
            patch_masks.append((y,x,masks.numpy())) 

            # argmax_masks = torch.argmax(masks, dim=0)
            # final_mask = torch.zeros(masks[0].shape)
            # final_mask_rescaled = torch.zeros(masks[0].shape).unsqueeze(-1).repeat(1,1,3)
        #save masks
        # for i in range(final_mask.shape[0]):
        #     for j in range(final_mask.shape[1]):
        #         final_mask[i,j] = codes[argmax_masks[i,j]] if masks[argmax_masks[i,j],i,j]>=0.5 else 0
        #         final_mask_rescaled[i,j] = torch.Tensor(visualize_dict[(labels_of_interest[argmax_masks[i,j]])] if masks[argmax_masks[i,j],i,j]>=0.5 else [0,0,0])

        #stitch masks
        print("original shape: ", original_img.shape)
        final_mask = stitch_patches_to_image(patch_masks, original_img.shape[:2])
        print("final mask shape: ",final_mask.shape)
        save_im = Image.fromarray(final_mask)
        save_im.save(os.path.join(args.save_path,'preds', img_name))

        # plt.imshow(final_mask_rescaled,cmap='gray')
        # plt.savefig(os.path.join(args.save_path,'rescaled_preds', img_name))
        # plt.close()
        break

if __name__ == '__main__':
    main()


        


