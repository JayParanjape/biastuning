from prompt_adapted_segment_anything.modeling.image_encoder import ImageEncoderViT
from prompt_adapted_segment_anything.modeling.mask_decoder import MaskDecoder
from prompt_adapted_segment_anything.modeling.prompt_encoder import PromptEncoder
from prompt_adapted_segment_anything.modeling import TwoWayTransformer
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Any, Dict, List, Tuple
import clip
from functools import partial, reduce
from operator import mul
import math
from typing import Union, List
from model_components.fdn import FDN

class Prompt_Adapted_SAM(nn.Module):
    def __init__(
        self, 
        config, 
        label_text_dict = {},
        device = 'cuda:0',
        training_strategy='biastuning'
        ):
        super().__init__()
        self.device = device
        self.img_size = config['sam']['img_size']
        self.num_classes = config['sam']['num_classes']
        self.label_dict = label_text_dict
        self.prompt_config = config['prompts']
        self.im_type = config['img_type']
        self.use_fdn = config['use_fdn']
        self.training_strategy = training_strategy

        #define hyperparameters, can be taken to a config later
        prompt_embed_dim=256
        image_embedding_size=16
        mask_in_chans=16

        print(self.prompt_config)
        #define pretrained clip and sam models
        self.sam_encoder = ImageEncoderViT(img_size=self.img_size,prompt_config=self.prompt_config)
        self.clip_model, _  = clip.load("ViT-B/32", device=device)

        #define the components of sam
        self.prompt_encoder=PromptEncoder(
        embed_dim=prompt_embed_dim,
        image_embedding_size=(image_embedding_size, image_embedding_size),
        input_image_size=(self.img_size, self.img_size),
        mask_in_chans=mask_in_chans,
        )

        self.mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
        transformer_dim=256,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
        )

        
        #define text prompt layers if they are to be used
        if self.prompt_config['USE_TEXT_PROMPT']:
            self.Text_Embedding_Affine = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256)
            )
            if self.training_strategy=='prompttuning':
                self.text_prompt_dropout = nn.Dropout(self.prompt_config['DROPOUT'])
                self.text_prompt_embeddings = nn.Parameter(torch.zeros(self.num_classes+1, prompt_embed_dim))
                nn.init.xavier_uniform_(self.text_prompt_embeddings.data)

                self.label_dict = self.label_dict.update({
                                        'other': self.num_classes
                                    })

        #define feature denormalization module if it is to be used
        if self.use_fdn:
            self.FDN_branch = FDN(norm_nc=256, input_nc=3, reduction_factor=4).to(device)

        #initialize sam with pretrained weights
        sam_ckpt = '/home/ubuntu/Desktop/Domain_Adaptation_Project/repos/segment-anything/checkpoints/sam_vit_b_01ec64.pth'
        # sam_ckpt = '/data/jparanj1/sam_vit_b_01ec64.pth'
        sam_state_dict = torch.load(sam_ckpt)
        for k in list(sam_state_dict.keys()):
            if self.img_size!=1024:
                #pos embed can be loaded only when image size is 1024
                if "pos_embed" in k:
                    full_matrix = sam_state_dict.pop(k)
                    adapted_matrix = nn.functional.adaptive_avg_pool2d(full_matrix.permute(0,3,1,2), (self.sam_encoder.pos_embed.shape[1], self.sam_encoder.pos_embed.shape[2]))
                    adapted_matrix = adapted_matrix.permute(0,2,3,1)
                    sam_state_dict[k] = adapted_matrix
            elif "image_encoder." in k:
                sam_state_dict[k[14:]] = sam_state_dict.pop(k)
            elif "prompt_encoder." in k:
                sam_state_dict[k[15:]] = sam_state_dict.pop(k)
            elif "mask_decoder." in k:
                sam_state_dict[k[13:]] = sam_state_dict.pop(k)


        self.sam_encoder.load_state_dict(sam_state_dict,strict=False)
        self.prompt_encoder.load_state_dict(sam_state_dict, strict=False)
        self.mask_decoder.load_state_dict(sam_state_dict,strict=False)

    def forward(self, x_img, x_text):
        B, C, H, W = x_img.shape
        x_text = list(x_text)
        
        if self.prompt_config['USE_TEXT_PROMPT']:
            if self.training_strategy=='prompttuning':
                prompt_text = []
                for t in x_text:
                    try:
                        prompt_text.append(self.text_prompt_embeddings[self.label_dict[t]])
                    except:
                        prompt_text.append(self.text_prompt_embeddings[-1])
                prompt_text = torch.stack(prompt_text)
        
        image_embeddings = self.sam_encoder(x_img)
        if self.use_fdn:
            image_embeddings = self.FDN_branch(image_embeddings, x_img)

        text_inputs = (clip.tokenize(x_text)).to(self.device)
        # with torch.no_grad():
        text_features = self.clip_model.encode_text(text_inputs)
            # text_features = text_features.unsqueeze(1)
        # print(text_features.shape)

        

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )

        # print(sparse_embeddings.shape)
        try:
            if self.prompt_config['USE_TEXT_PROMPT']:
                text_features_affine = self.Text_Embedding_Affine(text_features.float())
            else:
                text_features_affine = text_features[:,:256]
        except:
            print(text_features.shape)
            1/0
        if self.prompt_config['USE_TEXT_PROMPT'] and self.training_strategy=='prompttuning':
            text_features_affine = text_features_affine + prompt_text
        text_features_affine = text_features_affine.unsqueeze(1)
        sparse_embeddings = sparse_embeddings.to(self.device).repeat(B,1,1)
        sparse_embeddings = torch.cat(
            [sparse_embeddings,text_features_affine], dim=1)
        # print(sparse_embeddings.shape)
        # sparse_embeddings = sparse_embeddings.squeeze()
        # sparse_embeddings = sparse_embeddings.unsqueeze(1)

        low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                use_gsam = False
            )
        high_res_masks = self.postprocess_masks(low_res_masks, (self.img_size,self.img_size), (self.img_size,self.img_size))
        return high_res_masks

    def get_image_embeddings(self, x_img):
        with torch.no_grad():
            B, C, H, W = x_img.shape
            image_embeddings = self.sam_encoder(x_img)
            if self.use_fdn:
                image_embeddings = self.FDN_branch(image_embeddings, x_img)
            return image_embeddings

    def get_masks_for_multiple_labels(self, img_embeds, x_text):
        '''
        img_embeds - image embeddings obtained from get_imgae_embeddings function
        xtext - text prompts. image encoder wont be run and only the decoder will be run for each of these
        '''
        B = img_embeds.shape[0]
        with torch.no_grad():
            x_text = list(x_text)
            if self.prompt_config['USE_TEXT_PROMPT']:
                if self.training_strategy=='prompttuning':
                    prompt_text = []
                    for t in x_text:
                        try:
                            prompt_text.append(self.text_prompt_embeddings[self.label_dict[t]])
                        except:
                            prompt_text.append(self.text_prompt_embeddings[-1])
                    prompt_text = torch.stack(prompt_text)

            text_inputs = (clip.tokenize(x_text)).to(self.device)
            text_features = self.clip_model.encode_text(text_inputs)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                    points=None,
                    boxes=None,
                    masks=None,
                )

            if self.prompt_config['USE_TEXT_PROMPT']:
                text_features_affine = self.Text_Embedding_Affine(text_features.float())
            else:
                text_features_affine = text_features[:,:256]

            if self.prompt_config['USE_TEXT_PROMPT'] and self.training_strategy=='prompttuning':
                text_features_affine = text_features_affine + prompt_text
            
            text_features_affine = text_features_affine.unsqueeze(1)
            sparse_embeddings = sparse_embeddings.to(self.device).repeat(B,1,1)
            sparse_embeddings = torch.cat(
                [sparse_embeddings,text_features_affine], dim=1)

            low_res_masks, iou_predictions = self.mask_decoder(
                    image_embeddings=img_embeds,
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    use_gsam = False
                )
            high_res_masks = self.postprocess_masks(low_res_masks, (self.img_size,self.img_size), (self.img_size,self.img_size))
            return high_res_masks


    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.sam_encoder.img_size, self.sam_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        masks = torch.sigmoid(masks)
        return masks.squeeze(1)