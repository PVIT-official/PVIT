# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
#    Copyright 2023 Haotian Liu
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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM, \
                         CLIPVisionModel, CLIPImageProcessor

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .prompt_model import (
    DEFAULT_REGION_START_TOKEN, DEFAULT_REGION_END_TOKEN,
    SAMEncoder, PointDecoder, BoxDecoder
)
from .region_clip import (
    RegionCLIPPromptEncoder
)

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


class LlavaConfig(LlamaConfig):
    model_type = "llava"


class LlavaLlamaModel(LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig, mm_vision_tower=None, mm_hidden_size=None):
        super(LlavaLlamaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            # HACK: for FSDP
            self.vision_tower = [CLIPVisionModel.from_pretrained(config.mm_vision_tower)]

        if hasattr(config, "use_mm_proj"):
            self.mm_projector = nn.Linear(config.mm_hidden_size, config.hidden_size)
            
        if getattr(config, "mm_use_prompt_encoder", False):
            self.prompt_encoder = SAMEncoder(embedding_dim=1024)
            self.prompt_projector = nn.Linear(1024, config.hidden_size)
        
        if getattr(config, "mm_use_region_clip", None):
            self.prompt_projector = nn.Linear(640, self.config.hidden_size) # FIXME: hard coded
            
        if getattr(config, "mm_use_bbox_fc", False):
            self.bbox_fc = nn.Linear(4, self.config.hidden_size)

    def initialize_vision_modules(self, vision_tower, mm_vision_select_layer,
                                  pretrain_mm_mlp_adapter=None, tune_mm_mlp_adapter=False):
        self.config.mm_vision_tower = vision_tower

        image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        if not hasattr(self, 'vision_tower'):
            vision_tower = CLIPVisionModel.from_pretrained(vision_tower)
        else:
            vision_tower = self.vision_tower[0]
        vision_tower.requires_grad_(False)
        vision_tower = vision_tower.to(torch.float16)
        self.vision_tower = [vision_tower]

        vision_config = vision_tower.config
        num_patches = (vision_config.image_size // vision_config.patch_size) ** 2

        self.config.use_mm_proj = True
        self.config.mm_hidden_size = vision_config.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer

        if not hasattr(self, 'mm_projector'):
            self.mm_projector = nn.Linear(vision_config.hidden_size, self.config.hidden_size)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            self.mm_projector.load_state_dict({k.split('.')[-1]: v for k, v in mm_projector_weights.items()})
        
        return dict(
            image_processor=image_processor,
            image_token_len=num_patches,
            vision_config=vision_config
        )

    def initialize_prompt_modules(self, mm_use_prompt_encoder=False, mm_use_region_clip=None, mm_use_bbox_fc=False, dtype=torch.float16, device='cpu'):
        if not hasattr(self, "prompt_encoder"):
            if mm_use_prompt_encoder:
                self.prompt_encoder = SAMEncoder(embedding_dim=1024)
                if not hasattr(self, "prompt_projector"):
                    self.prompt_projector = nn.Linear(1024, self.config.hidden_size)
            if mm_use_region_clip:
                self.prompt_encoder = RegionCLIPPromptEncoder(weights=mm_use_region_clip, freeze=True)
                if not hasattr(self, "prompt_projector"):
                    self.prompt_projector = nn.Linear(self.prompt_encoder.hidden_size, self.config.hidden_size)
        if mm_use_bbox_fc:
            if not hasattr(self, "bbox_fc"):
                self.bbox_fc = nn.Linear(4, self.config.hidden_size)
        if mm_use_region_clip:
            self.prompt_encoder.region_encoder[0] = self.prompt_encoder.region_encoder[0].to(dtype=dtype, device=device)
                
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        boxes: Optional[List[torch.FloatTensor]] = None,
        region_clip_images = None,
        region_clip_boxes = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:

        # HACK: replace back original embeddings for LLaVA pretraining
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        vision_tower = getattr(self, 'vision_tower', None)
        if vision_tower is not None and (input_ids.shape[1] != 1 or self.training) and images is not None:
            # TODO: this is a modified multimodal LLM -- Haotian Liu
            vision_tower = vision_tower[0]  # HACK: for FSDP
            with torch.no_grad():
                if type(images) is list:
                    # variable length images
                    image_features = []
                    for image in images:
                        image_forward_out = vision_tower(image.unsqueeze(0), output_hidden_states=True)
                        select_hidden_state_layer = getattr(self.config, "mm_vision_select_layer", -1)
                        select_hidden_state = image_forward_out.hidden_states[select_hidden_state_layer]
                        image_feature = select_hidden_state[:, 1:]
                        image_features.append(image_feature)
                else:
                    image_forward_outs = vision_tower(images, output_hidden_states=True)
                    select_hidden_state_layer = getattr(self.config, "mm_vision_select_layer", -1)
                    select_hidden_state = image_forward_outs.hidden_states[select_hidden_state_layer]
                    image_features = select_hidden_state[:, 1:]
                
            if type(images) is list:
                image_features = [self.mm_projector(image_feature)[0] for image_feature in image_features]
            else:
                ori_image_features = image_features
                image_features = self.mm_projector(image_features)
            dummy_image_features = torch.zeros(256, 1024, device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            dummy_image_features = self.mm_projector(dummy_image_features)

            # use an additional prompt encoder to encode prompts
            if getattr(self.config, "mm_use_prompt_encoder", False):
                boxes_features = self.prompt_encoder(
                    image_embeddings=ori_image_features.detach(), # .detach() ?
                    boxes=boxes # List of N x 4
                ) # List of N x 2 x D 
                boxes_features = [self.prompt_projector(boxes_features_i) for boxes_features_i in boxes_features]
            elif getattr(self.config, "mm_use_region_clip", False) and region_clip_images is not None:
                boxes_features = self.prompt_encoder(
                    images=region_clip_images.to(inputs_embeds.device),
                    boxes=region_clip_boxes
                ) # List of N x D 
                boxes_features = [self.prompt_projector(boxes_features_i).unsqueeze(1) for boxes_features_i in boxes_features] # List of N x 1 x D 
                if hasattr(self, "bbox_fc"):
                    # Normalize RegionCLIP bbox
                    for i, (region_clip_image, region_clip_box) in enumerate(zip(region_clip_images, region_clip_boxes)):
                        h, w = region_clip_image.shape[1:]
                        region_clip_box_normalized = region_clip_box.tensor / torch.tensor([w, h, w, h])
                        box_position_features = self.bbox_fc(region_clip_box_normalized.to(boxes_features[0])).unsqueeze(1)
                        boxes_features[i] = boxes_features[i] + box_position_features
            else:
                boxes_features = None
            
            new_input_embeds = []
            cur_image_idx = 0
            for cur_idx, (cur_input_ids, cur_input_embeds) in enumerate(zip(input_ids, inputs_embeds)):
                if (cur_input_ids == vision_tower.config.im_patch_token).sum() == 0:
                    # multimodal LLM, but the current sample is not multimodal
                    cur_input_embeds = cur_input_embeds + (0. * dummy_image_features).sum()
                    new_input_embeds.append(cur_input_embeds)
                    continue
                if vision_tower.config.use_im_start_end:
                    cur_image_features = image_features[cur_image_idx]
                    num_patches = cur_image_features.shape[0]
                    if (cur_input_ids == vision_tower.config.im_start_token).sum() != (cur_input_ids == vision_tower.config.im_end_token).sum():
                        raise ValueError("The number of image start tokens and image end tokens should be the same.")
                    image_start_tokens = torch.where(cur_input_ids == vision_tower.config.im_start_token)[0]
                    for image_start_token_pos in image_start_tokens:
                        cur_image_features = image_features[cur_image_idx].to(device=cur_input_embeds.device)
                        num_patches = cur_image_features.shape[0]
                        if cur_input_ids[image_start_token_pos + num_patches + 1] != vision_tower.config.im_end_token:
                            raise ValueError("The image end token should follow the image start token.")
                        if orig_embeds_params is not None:
                            # LLM embeddings frozen
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:image_start_token_pos].detach(), cur_input_embeds[image_start_token_pos:image_start_token_pos+1], cur_image_features, cur_input_embeds[image_start_token_pos + num_patches + 1:image_start_token_pos + num_patches + 2], cur_input_embeds[image_start_token_pos + num_patches + 2:].detach()), dim=0)
                        else:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:image_start_token_pos+1], cur_image_features, cur_input_embeds[image_start_token_pos + num_patches + 1:]), dim=0)
                        cur_image_idx += 1
                    new_input_embeds.append(cur_new_input_embeds)
                else:
                    cur_image_features = image_features[cur_image_idx]
                    num_patches = cur_image_features.shape[0]
                    if (cur_input_ids == vision_tower.config.im_patch_token).sum() != num_patches:
                        raise ValueError("The number of image patch tokens should be the same as the number of image patches.")
                    masked_indices = torch.where(cur_input_ids == vision_tower.config.im_patch_token)[0]
                    mask_index_start = masked_indices[0]
                    if (masked_indices != torch.arange(mask_index_start, mask_index_start+num_patches, device=masked_indices.device, dtype=masked_indices.dtype)).any():
                        raise ValueError("The image patch tokens should be consecutive.")
                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start].detach(), cur_image_features, cur_input_embeds[mask_index_start+num_patches:].detach()), dim=0)
                    else:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], cur_image_features, cur_input_embeds[mask_index_start+num_patches:]), dim=0)
                    new_input_embeds.append(cur_new_input_embeds)
                
                if getattr(self.config, "mm_use_prompt_token", False) or getattr(self.config, "mm_use_prompt_encoder", False) or getattr(self.config, "mm_use_region_clip", None):
                    # [START] Region prompt embedding
                    if (cur_input_ids == self.config.region_start_token).sum() != (cur_input_ids == self.config.region_end_token).sum():
                        raise ValueError("The number of region start tokens and region end tokens should be the same.")
                    region_start_tokens = torch.where(cur_input_ids == self.config.region_start_token)[0]
                    cur_input_embeds = new_input_embeds[-1]
                    
                    if boxes_features is not None:
                        # use prompt encoder
                        cur_boxes_features = boxes_features[cur_idx] # N x 2 x D
                        if getattr(self.config, "mm_use_prompt_encoder", False):
                            num_region_token = 2 
                        if getattr(self.config, "mm_use_region_clip", None):
                            num_region_token = 1

                        if self.config.tune_mm_prompt_only:
                            cur_token_idx = 0
                            cur_new_input_embeds = []
                            for region_idx, region_start_token_pos in enumerate(region_start_tokens):
                                cur_new_input_embeds.append(cur_input_embeds[cur_token_idx:region_start_token_pos].detach()) # normal tokens
                                cur_token_idx = region_start_token_pos
                                
                                cur_new_input_embeds.append(cur_input_embeds[cur_token_idx:cur_token_idx+1]) # <Region>
                                cur_token_idx = cur_token_idx + 1
                                
                                cur_new_input_embeds.append(cur_boxes_features[region_idx].to(cur_input_embeds.device)) # <region_emb> <region_emb>
                                cur_token_idx = cur_token_idx + num_region_token
                                
                                cur_new_input_embeds.append(cur_input_embeds[cur_token_idx:cur_token_idx+1]) # </Region>
                                cur_token_idx = cur_token_idx + 1
                                
                            if cur_token_idx < len(cur_input_embeds):
                                cur_new_input_embeds.append(cur_input_embeds[cur_token_idx:].detach())
                            cur_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
                        else:
                            for region_idx, region_start_token_pos in enumerate(region_start_tokens):
                                cur_input_embeds[region_start_token_pos+1:region_start_token_pos+1+num_region_token] = cur_boxes_features[region_idx].to(cur_input_embeds.device) # <region_emb> <region_emb>
                        
                        new_input_embeds[-1] = cur_input_embeds
                    else:
                        # use prompt token
                        num_region_token = 6 # <Region> <x1> <y1> <x2> <y2> </Region>
                        if self.config.tune_mm_prompt_only:
                            cur_token_idx = 0
                            cur_new_input_embeds = []
                            for region_idx, region_start_token_pos in enumerate(region_start_tokens):
                                cur_new_input_embeds.append(cur_input_embeds[cur_token_idx:region_start_token_pos].detach())
                                cur_new_input_embeds.append(cur_input_embeds[region_start_token_pos:region_start_token_pos+num_region_token])
                                cur_token_idx = region_start_token_pos + num_region_token
                            
                            if cur_token_idx < len(cur_input_embeds):
                                cur_new_input_embeds.append(cur_input_embeds[cur_token_idx:].detach())
                            cur_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
                            new_input_embeds[-1] = cur_input_embeds
                    # [END] Region prompt embedding
                    
            inputs_embeds = torch.stack(new_input_embeds, dim=0)

        return super(LlavaLlamaModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


class LlavaLlamaForCausalLM(LlamaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if getattr(config, "mm_use_prompt_encoder", False):
            self.box_decoder = BoxDecoder(config.hidden_size, config.hidden_size, num_layers=3) # [..., D] -> [..., 4]
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        boxes: Optional[List[torch.FloatTensor]] = None,
        region_clip_images = None,
        region_clip_boxes = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            images=images,
            boxes=boxes,
            region_clip_images=region_clip_images,
            region_clip_boxes=region_clip_boxes,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

    def initialize_vision_tokenizer(self, mm_use_im_start_end, tokenizer, device,
                                    tune_mm_mlp_adapter=False, pretrain_mm_mlp_adapter=None):
        vision_config = self.model.vision_tower[0].config
        vision_config.use_im_start_end = mm_use_im_start_end
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)

        self.resize_token_embeddings(len(tokenizer))

        if mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if tune_mm_mlp_adapter:
                self.model.orig_embeds_params = [self.get_input_embeddings().weight.data.clone().to(device=device)]
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")

        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
    
    def initialize_prompt_tokenizer(self, tokenizer):
        old_tokenizer_len = len(tokenizer)
        tokenizer.add_tokens([DEFAULT_REGION_START_TOKEN, DEFAULT_REGION_END_TOKEN], special_tokens=True)
        
        if self.config.mm_use_prompt_token:
            tokenizer.add_tokens([f"<L{i}>" for i in range(1000)], special_tokens=True)
        
        new_tokenizer_len = len(tokenizer)
        self.resize_token_embeddings(new_tokenizer_len)
        num_new_tokens = new_tokenizer_len - old_tokenizer_len
        
        self.model.config.region_start_token, self.model.config.region_end_token = tokenizer.convert_tokens_to_ids(["<Region>", "</Region>"])
        self.config.region_start_token, self.config.region_end_token = self.model.config.region_start_token, self.model.config.region_end_token
        
        if num_new_tokens > 0:
            input_embeddings = self.get_input_embeddings().weight.data
            output_embeddings = self.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg
        
        if self.config.tune_mm_prompt_only:
            self.requires_grad_(False)
            for p in self.get_input_embeddings().parameters():
                p.requires_grad = True
            for p in self.get_output_embeddings().parameters():
                p.requires_grad = False

        if self.config.mm_use_prompt_encoder:
            if not hasattr(self, "box_decoder"):
                self.box_decoder = BoxDecoder(self.config.hidden_size, self.config.hidden_size, num_layers=3)
                
            for p in self.model.prompt_encoder.parameters():
                p.requires_grad = True
            for p in self.model.prompt_projector.parameters():
                p.requires_grad = True
            for p in self.box_decoder.parameters():
                p.requires_grad = True
        
        if self.config.mm_use_region_clip:
            for p in self.model.prompt_encoder.parameters():
                p.requires_grad = False
            for p in self.model.prompt_projector.parameters():
                p.requires_grad = True


AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
