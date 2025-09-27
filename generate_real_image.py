import sys
sys.path.append('..')


from transformers import CLIPTokenizer, T5TokenizerFast, logging
import os
import gc
import logging
from utils import util
from utils.flux_pipeline_real import FluxPipeline, calculate_shift, retrieve_timesteps
from utils.lora import LoRANetwork, DEFAULT_TARGET_REPLACE, UNET_TARGET_REPLACE_MODULE_CONV
import argparse
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxTransformer2DModel,
)
from diffusers import logging
from utils.flux_utils import *
from utils.compare_utils import *
import numpy as np
import random
import re

def comma_separated_numbers(value):
    return list(map(float, value.split(',')))

parser = argparse.ArgumentParser()
parser.add_argument("--lora_path", type=str)
parser.add_argument("--dataset", type=str)
parser.add_argument("--lora_name", type=str)
parser.add_argument("--use_prompts_file", type=int)
parser.add_argument("--use_random_seed", type=int)
parser.add_argument("--prompts_file", type=str)
parser.add_argument("--device", type=str)
parser.add_argument("--num_inference_steps", type=int)
parser.add_argument("--skip", type=int)
parser.add_argument("--start_seed", type=int)
parser.add_argument("--count", type=int)
parser.add_argument("--scales", type=comma_separated_numbers)
parser.add_argument("--output_path", type=str)
parser.add_argument("--guidance_scale", type=float)
parser.add_argument("--prompt", type=str)
parser.add_argument("--epoch", type=int)
parser.add_argument("--pretrained_model_name_or_path", type=str)
parser.add_argument("--semantic_lora_path", type=str)
parser.add_argument("--semantic_cfg_scale", type=float)

args = parser.parse_args()

lora_path = args.lora_path
dataset = args.dataset
lora_name = args.lora_name
prompts = util.load_prompts(args.prompts_file)
num_inference_steps = args.num_inference_steps
guidance_scale = args.guidance_scale
start_seed = args.start_seed
count = args.count
output_path = args.output_path
skip = args.skip
scales = args.scales
prompt = args.prompt
epoch = args.epoch
use_prompts_file = args.use_prompts_file
use_random_seed = args.use_random_seed
semantic_lora_path = args.semantic_lora_path
semantic_cfg_scale = args.semantic_cfg_scale

import os
os.environ["CUDA_VISIBLE_DEVICES"] = args.device
device = f"cuda:{args.device}"

import torch

logging.set_verbosity_warning()
logging.set_verbosity_error()
modules = DEFAULT_TARGET_REPLACE

pretrained_model_name_or_path = args.pretrained_model_name_or_path
weight_dtype = torch.bfloat16


max_sequence_length = 512
height = width = 512

# lora params
alpha = 1
rank = 16
train_method = 'full'

def flush():
    torch.cuda.empty_cache()
    gc.collect()
flush()


# Load the tokenizers
tokenizer_one = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer", torch_dtype=weight_dtype, device_map=device)
tokenizer_two = T5TokenizerFast.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer_2", torch_dtype=weight_dtype, device_map=device)

# Load scheduler and models
noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler", torch_dtype=weight_dtype, device_map=device)

# import correct text encoder classes
text_encoder_cls_one = import_model_class_from_model_name_or_path(pretrained_model_name_or_path, device=device)
text_encoder_cls_two = import_model_class_from_model_name_or_path(pretrained_model_name_or_path, subfolder="text_encoder_2", device=device)
# Load the text encoders
text_encoder_one, text_encoder_two = load_text_encoders(pretrained_model_name_or_path, text_encoder_cls_one, text_encoder_cls_two, weight_dtype, device)

# Load VAE
vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae", torch_dtype=weight_dtype)
transformer = FluxTransformer2DModel.from_pretrained(pretrained_model_name_or_path, subfolder="transformer", torch_dtype=weight_dtype)

# We only train the additional adapter LoRA layers
transformer.requires_grad_(False)
vae.requires_grad_(False)
text_encoder_one.requires_grad_(False)
text_encoder_two.requires_grad_(False)

vae.to(device)
transformer.to(device)
text_encoder_one.to(device)
text_encoder_two.to(device)
#transformer.enable_gradient_checkpointing()
print('Loaded Models')

tokenizers = [tokenizer_one, tokenizer_two]
text_encoders = [text_encoder_one, text_encoder_two]


# # lr_scheduler = {}
# params = []
modules = DEFAULT_TARGET_REPLACE
modules += UNET_TARGET_REPLACE_MODULE_CONV
network = LoRANetwork(
    transformer,
    rank=rank,
    multiplier=0.0,
    alpha=alpha,
    train_method=train_method,
).to(device, dtype=weight_dtype)
network.set_lora_slider(1)
network.load_state_dict(torch.load(f"{lora_path}"))

network_semantic = LoRANetwork(
    transformer,
    rank=rank,
    multiplier=0.0,
    alpha=alpha,
    train_method=train_method,
).to(device, dtype=weight_dtype)
network_semantic.load_state_dict(torch.load(f"{semantic_lora_path}"))
network_semantic.set_lora_slider(0)

pipe = FluxPipeline(noise_scheduler, vae, text_encoder_one, tokenizer_one, text_encoder_two, tokenizer_two, transformer,)
pipe.set_progress_bar_config(disable=False)


sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
image_seq_len = 1024
mu = calculate_shift(
    image_seq_len,
    pipe.scheduler.config.base_image_seq_len,
    pipe.scheduler.config.max_image_seq_len,
    pipe.scheduler.config.base_shift,
    pipe.scheduler.config.max_shift,
)
retrieve_timesteps(
    pipe.scheduler,
    num_inference_steps,
    device,
    None,
    sigmas,
    mu=mu,
)


# for seed in range(start_seed, start_seed + count):
for index in range(count): 
    if use_random_seed:
        seed = random.randint(0,2**15)
    else:
        seed = start_seed + index

    if use_prompts_file:
        prompt_index = random.randint(0, len(prompts) - 1)
        target_prompt = prompts[prompt_index]
    else:
        target_prompt = prompt
    

    seed_images = []
    slider_images = []

    scale_images = []
    origin_image = None

    single_image_path = f"{output_path}/origin"
    os.makedirs(single_image_path, exist_ok=True)

    prompt_file_name = re.sub(r'\s', '_', target_prompt)

    for slider_scale in scales:
        network_semantic.set_lora_slider(scale=slider_scale)
        image = pipe(
            target_prompt,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            max_sequence_length=max_sequence_length,
            num_images_per_prompt=1,
            generator=torch.Generator().manual_seed(seed),
            from_timestep=0,
            till_timestep=None,
            output_type='pil',
            network_content=network,
            network_semantic=network_semantic,
            cfg_scale=semantic_cfg_scale,
            skip_slider_timestep_till=skip, # this will skip adding the slider on the first step of generation ('1' will skip first 2 steps)
            device=device
        ).images[0]

        path = f"{single_image_path}/{seed}"
        os.makedirs(path, exist_ok=True)
        file_name = f"{path}/{seed}_{dataset}_skip{skip}_{epoch}epoch_scale{slider_scale:.2f}_{prompt_file_name}.jpg"
        print(f"Save:{file_name}")
        image.save(f"{file_name}")
        scale_images.append(image)

    if len(scales) > 1:
        scale_str = f"{scales[0]:.2f}-{scales[-1]:.2f}"
    else:
        scale_str = f"{scales[0]:.2f}"
    titles = [f"scale = {i}" if i !=0 else f"default(scale=0)" for i in scales]
    image_origin = create_collage(scale_images, titles, font_size=40)

    weight1 = lora_path.split("/")[-1]
    caption = f"seed={seed},skip={skip},prompt=\"{target_prompt}\"\nweight1={weight1}"
    create_collage_images_with_caption(image_origin, caption, f"{output_path}/{seed}_{dataset}_skip{skip}_{epoch}epoch_{scale_str}_{prompt_file_name}.jpg")