# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import re
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch
import torch.distributed

import transformers
from torch.utils.data import Dataset
from pvit.train.llava_trainer import LLaVATrainer

from pvit import conversation as conversation_lib
from pvit import LlavaLlamaForCausalLM

from pvit.data.multimodal_instruction import make_supervised_data_module

from PIL import Image
import torch.nn as nn

# TODO: import and use code from ../data/dataset.py

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_use_im_start_end: bool = field(default=False)
    
    mm_use_prompt_token: bool = field(default=False) # <Region><L101><L32><L64><L86></Region>
    mm_use_prompt_encoder: bool = field(default=False) # <Region><emb><emb></Region>
    mm_use_region_clip: Optional[str] = field(default=None) # path to regionclip ckpt
    mm_use_bbox_fc: bool = field(default=False) # add bbox positions to features
    tune_mm_prompt_only: bool = field(default=False)


@dataclass
class DataArguments:
    train_data_paths: List[str] = field(default=None,
                                        metadata={"help": "Paths to the training data."})
    train_image_folder: Optional[str] = field(default=None)
    eval_data_paths: List[str] = field(default=None,
                                       metadata={"help": "Paths to the evaluation data."})
    eval_image_folder: Optional[str] = field(default=None)
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_token_len: int = 0
    image_aspect_ratio: str = 'square'
    
    multitask_dataset: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    force_fsdp: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def prepare_vision_module(model, tokenizer, model_args, training_args, data_args=None):
    model_vision_dict = model.model.initialize_vision_modules(
            vision_tower=model_args.vision_tower,
            mm_vision_select_layer=model_args.mm_vision_select_layer,
            pretrain_mm_mlp_adapter=model_args.pretrain_mm_mlp_adapter
        )
    dtype = torch.float32
    if training_args.fp16:
        dtype = torch.float16
    if training_args.bf16:
        dtype = torch.bfloat16
    model.model.vision_tower[0].to(dtype=dtype, device=training_args.device)
    vision_config = model_vision_dict['vision_config']

    if data_args is not None:
        data_args.image_token_len = model_vision_dict['image_token_len']
        data_args.image_processor = model_vision_dict['image_processor']
        data_args.is_multimodal = True

    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    if model_args.tune_mm_mlp_adapter:
        model.requires_grad_(False)
        for p in model.model.mm_projector.parameters():
            p.requires_grad = True

    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
    if training_args.freeze_mm_mlp_adapter:
        for p in model.model.mm_projector.parameters():
            p.requires_grad = False

    model.config.mm_use_im_start_end = model_args.mm_use_im_start_end
    
    if data_args is not None:
        data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        data_args.mm_use_prompt_token = model_args.mm_use_prompt_token
        data_args.mm_use_prompt_encoder = model_args.mm_use_prompt_encoder
        data_args.mm_use_region_clip = model_args.mm_use_region_clip
        
    vision_config.use_im_start_end = training_args.use_im_start_end = model_args.mm_use_im_start_end
    model.initialize_vision_tokenizer(mm_use_im_start_end=model_args.mm_use_im_start_end, tokenizer=tokenizer, device=training_args.device,
                                        tune_mm_mlp_adapter=model_args.tune_mm_mlp_adapter, pretrain_mm_mlp_adapter=model_args.pretrain_mm_mlp_adapter)

    params_no_grad = [n for n, p in model.named_parameters() if not p.requires_grad]
    if len(params_no_grad) > 0:
        if training_args.fsdp is not None and len(training_args.fsdp) > 0:
            if len(params_no_grad) < 10:
                print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}'. format(len(params_no_grad), params_no_grad))
            else:
                print('[WARNING] Attempting to use FSDP while {} parameters do not require gradients: {}...(omitted)'. format(len(params_no_grad), ', '.join(params_no_grad[:10])))
            print("[WARNING] Attempting to use FSDP with partially frozen paramters, this is experimental.")
            print("[WARNING] As of 4/30/23, this feature requires PyTorch-nightly build.  See here for details: https://github.com/haotian-liu/LLaVA#experimental-use-fsdp-to-save-memory-in-pretraining")

            from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
            def patch_FSDP_use_orig_params(func):
                def wrap_func(*args, **kwargs):
                    use_orig_params = kwargs.pop('use_orig_params', True)
                    return func(*args, **kwargs, use_orig_params=use_orig_params)
                return wrap_func

            FSDP.__init__ = patch_FSDP_use_orig_params(FSDP.__init__)
    
    # prompt modules
    model.config.mm_use_prompt_token = model_args.mm_use_prompt_token
    model.config.mm_use_prompt_encoder = model_args.mm_use_prompt_encoder
    model.config.mm_use_region_clip = model_args.mm_use_region_clip
    model.config.mm_use_bbox_fc = model_args.mm_use_bbox_fc
    model.config.tune_mm_prompt_only = training_args.tune_mm_prompt_only = model_args.tune_mm_prompt_only
    
    model.model.initialize_prompt_modules(model_args.mm_use_prompt_encoder, model_args.mm_use_region_clip, model_args.mm_use_bbox_fc, dtype=dtype, device=training_args.device)
    model.initialize_prompt_tokenizer(tokenizer)
    
    if data_args is not None:
        return model, data_args
    else:
        return model


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.vision_tower is not None:
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=model,
            )
        if "llama" in model_args.model_name_or_path:
            tokenizer.add_special_tokens({
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            })
    else:
        tokenizer.pad_token = tokenizer.unk_token
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1_1"]

    if model_args.vision_tower is not None:
        model, data_args = prepare_vision_module(
            model, tokenizer, model_args, training_args, data_args)

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)

    if torch.distributed.get_rank() == 0:
        print("Trainable Parameters: ")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)
    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()
    safe_save_model_for_hf_trainer(trainer=trainer,
                                   output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
