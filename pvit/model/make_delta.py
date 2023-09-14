# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
"""
Usage:
python3 -m pvit.model.make_delta --base ~/model_weights/llama-7b --target ~/model_weights/pvit --delta ~/model_weights/pvit-delta
"""
import argparse

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from pvit.model.utils import auto_upgrade


def make_delta(base_model_path, target_model_path, delta_path):
    print("Loading base model")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)

    print("Loading target model")
    auto_upgrade(target_model_path)
    target = AutoModelForCausalLM.from_pretrained(target_model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True)

    print("Calculating delta")
    for name, param in tqdm(target.state_dict().items(), desc="Calculating delta"):
        if name not in base.state_dict():
            assert name in [
                'model.mm_projector.weight',
                'model.mm_projector.bias',
                'model.prompt_projector.weight',
                'model.prompt_projector.bias',
                'model.bbox_fc.weight',
                'model.bbox_fc.bias',
            ], f'{name} not in base model'
            continue
        if param.data.shape == base.state_dict()[name].shape:
            param.data -= base.state_dict()[name]
        else:
            assert name in ['model.embed_tokens.weight', 'lm_head.weight'], f'{name} dimension mismatch: {param.data.shape} vs {base.state_dict()[name].shape}'
            bparam = base.state_dict()[name]
            param.data[:bparam.shape[0], :bparam.shape[1]] -= bparam

    print("Saving delta")
    target.save_pretrained(delta_path)
    target_tokenizer = AutoTokenizer.from_pretrained(target_model_path)
    target_tokenizer.save_pretrained(delta_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--target-model-path", type=str, required=True)
    parser.add_argument("--delta-path", type=str, required=True)
    args = parser.parse_args()

    make_delta(args.base_model_path, args.target_model_path, args.delta_path)
