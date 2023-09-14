import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import time
from typing import Dict, List, Optional, Tuple

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.structures import ImageList, Boxes, Instances

import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T
from detectron2.modeling.meta_arch.clip_rcnn import visualize_proposals

def get_regionclip_cfg():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, 'region_clip_config.yaml')
    
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    return cfg

def get_region_clip_transforms():
    cfg = get_regionclip_cfg()
    # image transformation
    augs = utils.build_augmentation(cfg, False)
    augmentations = T.AugmentationList(augs)
    
    div_pixel = True
    pixel_mean, pixel_std = torch.tensor(cfg.MODEL.PIXEL_MEAN).reshape(-1, 1, 1), torch.tensor(cfg.MODEL.PIXEL_STD).reshape(-1, 1, 1)
    
    def _transforms(image, boxes=None):
        aug_input = T.AugInput(image, boxes=boxes)
        transforms = augmentations(aug_input)
        image = aug_input.image
        image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        
        """
        Normalize, pad and batch the input images. Use CLIP default processing (pixel mean & std).
        Note: Due to FPN size_divisibility, images are padded by right/bottom border. So FPN is consistent with C4 and GT boxes.
        """
        if div_pixel:
            image = ((image / 255.0) - pixel_mean) / pixel_std
        else:
            image = (image - pixel_mean) / pixel_std
        
        if boxes is not None:
            boxes = aug_input.boxes
            return image, Boxes(torch.from_numpy(boxes))
        else:
            return image
    
    return _transforms
    
def create_region_clip(weights='regionclip_pretrained-cc_rn50x4.pth', freeze=True):
    cfg = get_regionclip_cfg()
    cfg.MODEL.CLIP.CROP_REGION_TYPE = "GT"
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.DEVICE = 'cpu'
    
    # create model
    model = DefaultTrainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=False
    )
    if cfg.MODEL.META_ARCHITECTURE in ['CLIPRCNN', 'CLIPFastRCNN', 'PretrainFastRCNN'] \
        and cfg.MODEL.CLIP.BB_RPN_WEIGHTS is not None\
        and cfg.MODEL.CLIP.CROP_REGION_TYPE == 'RPN': # load 2nd pretrained model
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR, bb_rpn_weights=True).resume_or_load(
            cfg.MODEL.CLIP.BB_RPN_WEIGHTS, resume=False
        )
    
    assert model.use_clip_c4 # use C4 + resnet weights from CLIP
    assert model.use_clip_attpool # use att_pool from CLIP to match dimension
    
    if freeze:
        for p in model.parameters(): 
            p.requires_grad = False
    model.eval()
    return model, cfg

def extract_region_feats(model, images, proposal_boxes):
    """ Given a model and the input images, extract region features and save detection outputs into a local file
    (refer to detectron2/modeling/meta_arch/clip_rcnn.py)
    """
    # model inference
    # 1. recognition branch: get 2D feature maps using the backbone of recognition branch
    features = model.backbone(images.tensor)

    # 2. given the proposals, crop region features from 2D image features
    box_features = model.roi_heads._shared_roi_transform(
        [features[f] for f in model.roi_heads.in_features], proposal_boxes, model.backbone.layer4
    )
    att_feats = model.backbone.attnpool(box_features)  # region features
    return att_feats

class RegionCLIPPromptEncoder(nn.Module):
    
    def __init__(self, weights='regionclip_pretrained-cc_rn50x4.pth', freeze=True, dtype=torch.float16, device='cpu'):
        super().__init__()
        
        region_encoder, self.config = create_region_clip(weights, freeze)
        # HACK: for FSDP
        region_encoder = region_encoder.to(dtype=dtype, device=device)
        self.region_encoder = [region_encoder]
        self.hidden_size = self.config.MODEL.CLIP.TEXT_EMB_DIM
    
    def forward(
        self, 
        images: ImageList,
        boxes: List[Boxes]
    ):
        features = self.region_encoder[0].backbone(images.tensor)

        boxes = [boxes_i.to(images.tensor.device) for boxes_i in boxes]
        box_features = self.region_encoder[0].roi_heads._shared_roi_transform(
            [features[f] for f in self.region_encoder[0].roi_heads.in_features], boxes, self.region_encoder[0].backbone.layer4
        )
        
        att_feats = self.region_encoder[0].backbone.attnpool(box_features)  # region features
        region_feats = torch.split(att_feats, [len(boxes_i) for boxes_i in boxes], dim=0)
        return region_feats
