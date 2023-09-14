from .llava import (
    LlavaLlamaForCausalLM,
    DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN, 
    DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
)

from .prompt_model import (
    DEFAULT_REGION_PATCH_TOKEN, DEFAULT_REGION_START_TOKEN, DEFAULT_REGION_END_TOKEN,
    convert_from_prompt_tokens
)
from .region_clip import (
    RegionCLIPPromptEncoder
)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer

def load_model(model_path, num_gpus, dtype=torch.float16, max_memory='13GiB'):
    if num_gpus == 0:
        kwargs = {
            "device_map": "cpu"
        }
    if num_gpus == 1:
        kwargs = {}
    else:
        kwargs = {"device_map": "auto"}

    print(kwargs)

    tokenizer = LlamaTokenizer.from_pretrained(model_path) 
    if 'llava' in model_path.lower():
        model, loading_info = LlavaLlamaForCausalLM.from_pretrained(model_path, torch_dtype=dtype, low_cpu_mem_usage=True, output_loading_info=True, **kwargs)
    else:
        model, loading_info = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, low_cpu_mem_usage=True, output_loading_info=True, **kwargs)
    
    print(loading_info)
    image_processor = None

    if hasattr(model.config, "mm_vision_tower"):
        from transformers import CLIPImageProcessor, CLIPVisionModel
        image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=dtype)

        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_prompt_token = getattr(model.config, "mm_use_prompt_token", False)
        mm_use_region_clip = getattr(model.config, "mm_use_region_clip", False)
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        if mm_use_prompt_token or mm_use_region_clip:
            tokenizer.add_tokens([DEFAULT_REGION_START_TOKEN, DEFAULT_REGION_END_TOKEN], special_tokens=True)
        if mm_use_prompt_token:
            tokenizer.add_tokens([f"<L{i}>" for i in range(1000)], special_tokens=True)

        vision_tower = model.model.vision_tower[0]
        if vision_tower.device.type == 'meta':
            vision_tower = CLIPVisionModel.from_pretrained(vision_tower.config._name_or_path, torch_dtype=dtype, low_cpu_mem_usage=True).cuda()
            model.model.vision_tower[0] = vision_tower
        else:
            vision_tower.to(device='cuda', dtype=dtype)
        vision_config = vision_tower.config
        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = mm_use_im_start_end
        if mm_use_im_start_end:
            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

        if getattr(model.config, "mm_use_region_clip", None):
            model.model.prompt_encoder = RegionCLIPPromptEncoder(model.config.mm_use_region_clip, freeze=True, dtype=dtype, device='cuda')

    if num_gpus == 1:
        model.cuda()

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len