# Adopted from https://github.com/haotian-liu/LLaVA. Below is the original copyright:
"""
A model worker executes the model.
"""
import re
import argparse
import asyncio
import dataclasses
import logging
import json
import time
import numpy as np
from typing import List, Union
import threading
import uuid
import traceback

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import uvicorn
from functools import partial

from pvit.constants import WORKER_HEART_BEAT_INTERVAL
from pvit.utils import (build_logger, server_error_msg,
    pretty_print_semaphore)
from pvit.model import (
    DEFAULT_IMAGE_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
    load_model, convert_from_prompt_tokens
)
from pvit.data.transforms import clip_image_transform, region_clip_image_transform
from detectron2.structures import ImageList, Boxes

GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger("model_worker", f"model_worker_{worker_id}.log")
global_counter = 0

model_semaphore = None

DEFAULT_DTYPE = torch.float16 

def heart_beat_worker(controller):

    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()

region_pattern = r'<Region>(\s*<L(\d{1,4})>\s*<L(\d{1,4})>\s*<L(\d{1,4})>\s*<L(\d{1,4})>\s*)</Region>'

class ModelWorker:
    def __init__(self, controller_addr, worker_addr,
                 worker_id, no_register,
                 model_path, model_name,
                 keep_aspect_ratio,
                 num_gpus):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        if model_path.endswith("/"):
            model_path = model_path[:-1]
        if model_name is None:
            model_paths = model_path.split("/")
            if model_paths[-1].startswith('checkpoint-'):
                self.model_name = model_paths[-2] + "_" + model_paths[-1]
            else:
                self.model_name = model_paths[-1]
        else:
            self.model_name = model_name

        logger.info(f"Loading the model {self.model_name} on worker {worker_id} ...")
        self.keep_aspect_ratio = keep_aspect_ratio
        self.tokenizer, self.model, self.image_processor, self.context_len = load_model(
            model_path, num_gpus, dtype=DEFAULT_DTYPE)
        self.is_multimodal = True

        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,))
            self.heart_beat_thread.start()

    def register_to_controller(self):
        logger.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status()
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200

    def send_heart_beat(self):
        logger.info(f"Send heart beat. Models: {[self.model_name]}. "
                    f"Semaphore: {pretty_print_semaphore(model_semaphore)}. "
                    f"global_counter: {global_counter}")

        url = self.controller_addr + "/receive_heart_beat"

        while True:
            try:
                ret = requests.post(url, json={
                    "worker_name": self.worker_addr,
                    "queue_length": self.get_queue_length()}, timeout=5)
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"heart beat error: {e}")
            time.sleep(5)

        if not exist:
            self.register_to_controller()

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return args.limit_model_concurrency - model_semaphore._value + (len(
                model_semaphore._waiters) if model_semaphore._waiters is not None else 0)

    def get_status(self):
        return {
            "model_names": [self.model_name],
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    @torch.inference_mode()
    def generate_stream(self, params):
        tokenizer, model, image_processor = self.tokenizer, self.model, self.image_processor

        prompt = params["prompt"]
        ori_prompt = prompt
        images = params.get("images", None)
        if images is not None and self.is_multimodal:
            from PIL import Image
            from io import BytesIO
            import base64
            assert type(images) is list
            multimodal_prompt_args = {}
            if len(images) > 0:
                images = [Image.open(BytesIO(base64.b64decode(image))).convert('RGB') for image in images]
                assert len(images) == prompt.count(DEFAULT_IMAGE_TOKEN), "Number of images does not match number of <image> tokens in prompt"

                if self.keep_aspect_ratio:
                    new_images = []
                    for image_idx, image in enumerate(images):
                        max_hw, min_hw = max(image.size), min(image.size)
                        aspect_ratio = max_hw / min_hw
                        max_len, min_len = 448, 224
                        shortest_edge = int(min(max_len / aspect_ratio, min_len))
                        image = image_processor.preprocess(image, return_tensors='pt', do_center_crop=False, size={"shortest_edge": shortest_edge})['pixel_values'][0]
                        new_images.append(image.to(self.model.device, dtype=DEFAULT_DTYPE))
                        # replace the image token with the image patch token in the prompt (each occurrence)
                        cur_token_len = (image.shape[1]//14) * (image.shape[2]//14)
                        replace_token = DEFAULT_IMAGE_PATCH_TOKEN * cur_token_len
                        if getattr(self.model.config, 'mm_use_im_start_end', False):
                            replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token, 1)
                    clip_images = new_images
                else:
                    import torchshow
                    clip_images = torch.stack([clip_image_transform(image) for image in images])
                    clip_images = clip_images.to(self.model.device, dtype=DEFAULT_DTYPE)
                    replace_token = DEFAULT_IMAGE_PATCH_TOKEN * 256    # HACK: 256 is the max image token length hacked
                    if getattr(self.model.config, 'mm_use_im_start_end', False):
                        replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                    prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
                    
                    boxes = None
                    if getattr(model.config, "mm_use_prompt_encoder", False):
                        prompt, boxes = convert_from_prompt_tokens(prompt, num_region_token=2)
                        if boxes is not None:
                            multimodal_prompt_args["boxes"] = [torch.tensor(boxes).to(self.model.device, dtype=DEFAULT_DTYPE)]
                    if getattr(model.config, "mm_use_region_clip", False):
                        prompt, boxes = convert_from_prompt_tokens(prompt, num_region_token=1)
                        if len(boxes):
                            image = images[0]
                            width, height = image.size
                            boxes = np.array(boxes) * np.array([width, height, width, height])
                            region_clip_image, region_clip_boxes = region_clip_image_transform(np.asarray(image), boxes)
                            multimodal_prompt_args["boxes"] = [torch.tensor(boxes).to(self.model.device, dtype=DEFAULT_DTYPE)]
                            multimodal_prompt_args['region_clip_images'] = ImageList.from_tensors([region_clip_image])
                            multimodal_prompt_args['region_clip_boxes'] = [region_clip_boxes] 
            else:
                clip_images = None
                boxes = None
            image_args = {"images": clip_images}
        else:
            images = None
            image_args = {}
            multimodal_prompt_args = {}

        l_prompt = len(prompt)
        temperature = float(params.get("temperature", 1.0))
        max_new_tokens = min(int(params.get("max_new_tokens", 256)), 1024)
        stop_str = params.get("stop", None)

        input_ids = tokenizer(prompt).input_ids
        output_ids = list(input_ids)
        pred_ids = []

        max_src_len = self.context_len - max_new_tokens - 8
        input_ids = input_ids[-max_src_len:]

        past_key_values = None
        for i in range(max_new_tokens):
            if i == 0:
                out = model(
                    torch.as_tensor([input_ids]).cuda(),
                    use_cache=True,
                    **image_args,
                    **multimodal_prompt_args)
                logits = out.logits
                past_key_values = out.past_key_values
            else:
                attention_mask = torch.ones(
                    1, past_key_values[0][0].shape[-2] + 1, device="cuda")
                out = model(input_ids=torch.as_tensor([[token]], device="cuda"),
                            use_cache=True,
                            attention_mask=attention_mask,
                            past_key_values=past_key_values)
                logits = out.logits
                past_key_values = out.past_key_values

            last_token_logits = logits[0][-1]
            if temperature < 1e-4:
                token = int(torch.argmax(last_token_logits))
            else:
                probs = torch.softmax(last_token_logits / temperature, dim=-1)
                token = int(torch.multinomial(probs, num_samples=1))

            output_ids.append(token)
            pred_ids.append(token)

            if token == tokenizer.eos_token_id:
                stopped = True
            else:
                stopped = False

            if i % args.stream_interval == 0 or i == max_new_tokens - 1 or stopped:
                cur_out = tokenizer.decode(pred_ids, skip_special_tokens=True)
                pos = cur_out.rfind(stop_str)
                if pos != -1:
                    cur_out = cur_out[:pos]
                    stopped = True
                output = ori_prompt + cur_out

                ret = {
                    "text": output,
                    "error_code": 0,
                }
                yield json.dumps(ret).encode() + b"\0"

            if stopped:
                break

        if past_key_values is not None:
            del past_key_values

    def generate_stream_gate(self, params):
        try:
            for x in self.generate_stream(params):
                yield x
        except ValueError as e:
            print("Caught ValueError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            traceback.print_exc()
            yield json.dumps(ret).encode() + b"\0"
        except torch.cuda.CudaError as e:
            print("Caught torch.cuda.CudaError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"


app = FastAPI()


def release_model_semaphore(fn=None):
    model_semaphore.release()
    if fn is not None:
        fn()


@app.post("/worker_generate_stream")
async def generate_stream(request: Request):
    global model_semaphore, global_counter
    global_counter += 1
    params = await request.json()

    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    await model_semaphore.acquire()
    worker.send_heart_beat()
    generator = worker.generate_stream_gate(params)
    background_tasks = BackgroundTasks()
    background_tasks.add_task(partial(release_model_semaphore, fn=worker.send_heart_beat))
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_get_status")
async def get_status(request: Request):
    return worker.get_status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=21002)
    parser.add_argument("--worker-address", type=str,
        default="http://localhost:21002")
    parser.add_argument("--controller-address", type=str,
        default="http://localhost:21001")
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--multi-modal", action="store_true", help="Multimodal mode is automatically detected with model name, please make sure `llava` is included in the model path.")
    parser.add_argument("--keep-aspect-ratio", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    if args.multi_modal:
        logger.warning("Multimodal mode is automatically detected with model name, please make sure `llava` is included in the model path.")

    worker = ModelWorker(args.controller_address,
                         args.worker_address,
                         worker_id,
                         args.no_register,
                         args.model_path,
                         args.model_name,
                         args.keep_aspect_ratio,
                         args.num_gpus)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
