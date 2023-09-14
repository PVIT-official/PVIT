import os
import copy
import glob
import json
import logging
from typing import Dict, Optional, Sequence, List
import torch
import yaml
import random
import numpy as np
from dataclasses import dataclass
from collections import defaultdict

from PIL import Image
import transformers
from torch.utils.data import Dataset

from pvit import conversation as conversation_lib
from pvit.model import (
    DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN, 
    DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
    convert_from_prompt_tokens
)
from pvit.data.transforms import clip_image_transform, region_clip_image_transform

from detectron2.structures import ImageList, Boxes

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"

def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation

def preprocess_multimodal(
    sources: Sequence[str],
    multimodal_cfg: dict,
    cur_token_len: int,
) -> Dict:
    is_multimodal = multimodal_cfg['is_multimodal']
    image_token_len = cur_token_len
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            replace_token = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
            if multimodal_cfg['use_im_start_end']:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
            
            if multimodal_cfg['use_prompt_encoder']:
                sentence["value"], sentence["boxes"] = convert_from_prompt_tokens(sentence["value"], num_region_token=2)
            if multimodal_cfg['use_region_clip']:
                sentence["value"], sentence["boxes"] = convert_from_prompt_tokens(sentence["value"], num_region_token=1)

    return sources

def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    boxes = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        boxes_i = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            
            conv.append_message(role, sentence["value"])
            if "boxes" in sentence:
                boxes_i.extend(sentence["boxes"]) # N x 4
        conversations.append(conv.get_prompt())
        if len(boxes_i):
            boxes.append(boxes_i)
        else:
            boxes.append(None)

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer(rou).input_ids)
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
        boxes=boxes
    )

def preprocess_v1_label_only(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    label_start_token: str = '<CLabel>',
    label_end_token: str = '</CLabel>'
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    boxes = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        boxes_i = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
            if "boxes" in sentence:
                boxes_i.extend(sentence["boxes"]) # N x 4
        conversations.append(conv.get_prompt())
        if len(boxes_i):
            boxes.append(boxes_i)
        else:
            boxes.append(None)

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    
    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO
    
    if label_start_token is not None:
        assert label_end_token is not None
        
        label_start_token_id, label_end_token_id = \
            tokenizer.convert_tokens_to_ids(label_start_token), \
            tokenizer.convert_tokens_to_ids(label_end_token)
        
        new_input_ids = tokenizer(
            [conv.replace(f'{label_start_token} ', '').replace(f' {label_end_token}', '') for conv in conversations],
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
        
        targets = torch.full_like(new_input_ids, IGNORE_INDEX)
        
        for input_ids_i, new_input_ids_i, target in zip(input_ids, new_input_ids, targets):
            label_start_pos = torch.where(input_ids_i == label_start_token_id)[0]
            label_end_pos = torch.where(input_ids_i == label_end_token_id)[0]

            label_start_pos = label_start_pos - torch.arange(len(label_start_pos)) * 2
            label_end_pos = label_end_pos - torch.arange(len(label_end_pos)) * 2 - 1

            for start, end in zip(label_start_pos, label_end_pos):
                target[start: end] = new_input_ids_i[start: end]
        
        input_ids = new_input_ids
        
    else:    
        targets = input_ids.clone()

        # Mask targets
        sep = conv.sep + conv.roles[1] + ": "
        for conversation, target in zip(conversations, targets):
            total_len = int(target.ne(tokenizer.pad_token_id).sum())

            rounds = conversation.split(conv.sep2)
            cur_len = 1
            target[:cur_len] = IGNORE_INDEX
            for i, rou in enumerate(rounds):
                if rou == "":
                    break

                parts = rou.split(sep)
                if len(parts) != 2:
                    break
                parts[0] += sep
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

                cur_len += round_len
            target[cur_len:] = IGNORE_INDEX

            if cur_len < tokenizer.model_max_length:
                if cur_len != total_len:
                    target[:] = IGNORE_INDEX
                    print(
                        f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                        f" (ignored)"
                    )

    return dict(
        input_ids=input_ids,
        labels=targets,
        boxes=boxes
    )

def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.version == "v1":
        return preprocess_v1(sources, tokenizer)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    conversations_tokenized = _tokenize_fn(conversations, tokenizer)
    input_ids = conversations_tokenized["input_ids"]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source],
                                      tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)

def parse_data_mapping(mapping_file):
    data_mapping = yaml.load(open(mapping_file), Loader=yaml.Loader)
    image_paths = data_mapping['image_paths']
    mapping = data_mapping['mapping']
    has_bbox = data_mapping.get('bbox', None)
    
    res = {}
    for data_key in mapping:
        if has_bbox is None or has_bbox[data_key] == 'input': # input, output, neither, both
            res[data_key] = {
                'image_path': image_paths[mapping[data_key]],
            }
    return res

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
        
        if 'boxes' in instances[0]:
            batch['boxes'] = [torch.tensor(instance['boxes']) if instance['boxes'] is not None else None for instance in instances]
        
        if 'region_clip_image' in instances[0]:
            batch['region_clip_images'] = ImageList.from_tensors([instance['region_clip_image'] for instance in instances])
            
        if 'region_clip_boxes' in instances[0]:
            batch['region_clip_boxes'] = [instance['region_clip_boxes'] for instance in instances]

        return batch

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    
    multimodal_cfg=dict(is_multimodal=data_args.is_multimodal,
                        image_token_len=data_args.image_token_len,
                        image_aspect_ratio=data_args.image_aspect_ratio,
                        use_im_start_end=getattr(data_args, 'mm_use_im_start_end', False),
                        use_prompt_token=getattr(data_args, 'mm_use_prompt_token', False),
                        use_prompt_encoder=getattr(data_args, 'mm_use_prompt_encoder', False),
                        use_region_clip=getattr(data_args, 'mm_use_region_clip', False),
                        image_processor=getattr(data_args, 'image_processor', None))
    
    if os.path.isdir(data_args.train_data_paths[0]):
        dataset_cls = MultiSourceImageDataset
    else:
        dataset_cls = LazySupervisedDataset
        
    train_multimodal_cfg = copy.deepcopy(multimodal_cfg)
    train_multimodal_cfg['image_folder'] = data_args.train_image_folder
    train_dataset = dataset_cls(tokenizer=tokenizer,
                                data_paths=data_args.train_data_paths,
                                multimodal_cfg=train_multimodal_cfg)
    eval_dataset = None
    if data_args.eval_data_paths is not None:
        eval_multimodal_cfg = copy.deepcopy(multimodal_cfg)
        eval_multimodal_cfg['image_folder'] = data_args.eval_image_folder
        eval_dataset = dataset_cls(tokenizer=tokenizer,
                                    data_paths=data_args.eval_data_paths,
                                    multimodal_cfg=eval_multimodal_cfg)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator)

import re
def perturb_bbox_str(s_with_regions, image_width, image_height, box_format='xywh', max_shift_percent=0.02):
    REGION_PATTERN = r'\[(\d+(\.\d+)?),\s*(\d+(\.\d+)?),\s*(\d+(\.\d+)?),\s*(\d+(\.\d+)?)\]'
  
    boxes_strs = re.finditer(REGION_PATTERN, s_with_regions)
    for match in boxes_strs:
        box_str = match.group(0)
        box = eval(box_str)

        box = np.array(box).reshape(1, -1) * np.array([image_width, image_height, image_width, image_height]) # 1 x 4
        box = perturb_bbox(box, image_width, image_height) / np.array([image_width, image_height, image_width, image_height]) # 4
        box = box[0]
        new_box_str = '[' + ','.join([str(int(x * 1000) / 1000) for x in box]) + ']'
      
        s_with_regions = s_with_regions.replace(box_str, new_box_str)
    return s_with_regions

def perturb_bbox(bbox, image_width, image_height, max_shift_percent=0.02):
    """
    bbox perturbation

    args:
    - bbox: numpy array, shape of(N, 4), coordinate of bbox, formatted as xyxy
    - max_shift_percent: float, maximum percentage of perturbation
    - image_width: int
    - image_height: int

    return values:
    - perturbed_bbox: numpy array, shape of(N, 4), coordinate after perturbation
    """

    widths = bbox[:, 2] - bbox[:, 0]
    heights = bbox[:, 3] - bbox[:, 1]

    shifts_percent = np.random.uniform(-max_shift_percent, max_shift_percent, size=bbox.shape)

    shifts = shifts_percent * np.stack([widths, heights, widths, heights], axis=1)

    perturbed_bbox = bbox + shifts

    perturbed_bbox[:, 0] = np.clip(perturbed_bbox[:, 0], 0, image_width)
    perturbed_bbox[:, 1] = np.clip(perturbed_bbox[:, 1], 0, image_height)
    perturbed_bbox[:, 2] = np.clip(perturbed_bbox[:, 2], 0, image_width)
    perturbed_bbox[:, 3] = np.clip(perturbed_bbox[:, 3], 0, image_height)

    return perturbed_bbox


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_paths: List[str],
                 tokenizer: transformers.PreTrainedTokenizer,
                 multimodal_cfg: dict):
        super(LazySupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = []
        for data_path in data_paths:
            list_data_dict.extend(json.load(open(data_path, "r")))

        logging.warning("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.multimodal_cfg = multimodal_cfg

    def __len__(self):
        return len(self.list_data_dict)

    def get_image(self, i):
        image_file = self.list_data_dict[i]['image']
        image_folder = self.multimodal_cfg['image_folder']
        image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
        return image

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        
        # image exist in the data
        image = None
        if 'image' in sources[0]:
            image = self.get_image(i)
            width, height = image.size
            clip_image = clip_image_transform(image)
            cur_token_len = (clip_image.shape[1]//14) * (clip_image.shape[2]//14)   # FIXME: 14 is hardcoded patch size
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.multimodal_cfg, cur_token_len)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        
        for source in sources:
            for sentence in source:
                sentence['value'] = perturb_bbox_str(sentence['value'], width, height)

        data_dict = preprocess(
            sources,
            self.tokenizer)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0],
                             boxes=data_dict["boxes"][0])
            
        if image is not None:
            data_dict['image'] = clip_image
        elif self.multimodal_cfg['is_multimodal']:
            crop_size = self.multimodal_cfg['image_processor'].crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            
        # image exist in the data
        if image is not None and self.multimodal_cfg['use_region_clip']:
            if data_dict['boxes']:
                width, height = image.size
                boxes = np.array(data_dict['boxes']) * np.array([width, height, width, height])
                # Add box shift
                boxes = perturb_bbox(boxes, width, height)
            else:
                # dummy bounding boxes
                width, height = image.size
                boxes = np.array([[0, 0, 1, 1]]) * np.array([width, height, width, height])
            region_clip_image, region_clip_boxes = region_clip_image_transform(np.asarray(image), boxes)
            data_dict['region_clip_image'], data_dict['region_clip_boxes'] = region_clip_image, region_clip_boxes
        
        return data_dict

class MultiSourceImageDataset(LazySupervisedDataset):
    
    def __init__(self, 
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_paths: str,
                 multimodal_cfg: dict):
        if isinstance(data_paths, str):
            data_dir = data_paths
        else:
            data_dir = data_paths[0]
        assert os.path.isdir(data_dir)
        
        list_data_dict = []
        list_image_folder = []
        datasets = parse_data_mapping(f"{data_dir}/mapping.yaml") # image_path
        
        # sampling
        sample_ratio = datasets.get('sample_ratio', {})
        
        for key in datasets:
            data_file = f"{data_dir}/{key}.json"
            data = json.load(open(data_file, "r"))
            if key in sample_ratio:
                n_samples = int(len(data) * sample_ratio[key])
                data = random.choices(data, k=n_samples)
            
            list_data_dict.extend(data)
            list_image_folder.extend([datasets[key]['image_path'] for _ in data])
        
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.list_image_folder = list_image_folder
        self.multimodal_cfg = multimodal_cfg

    def get_image(self, i):
        image_file = self.list_data_dict[i]['image']
        image_folder = self.list_image_folder[i]
        image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
        return image
    

def capitalize_parts(string):
    parts = string.split('_')
    capitalized_parts = [part.capitalize() for part in parts]
    result = '_'.join(capitalized_parts)
    return result
