import sys
sys.path.append('..')
import argparse

import argparse
import gc
import logging
from tqdm.auto import tqdm
from transformers import CLIPTokenizer, T5TokenizerFast
from utils.compare_utils import *
from utils.flux_utils import *
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxTransformer2DModel
from diffusers.optimization import get_scheduler

from utils.flux_pipeline import FluxPipeline, calculate_shift, retrieve_timesteps
import argparse
import gc
import torch
from tqdm.auto import tqdm
import numpy as np
from torch.optim import AdamW
from contextlib import ExitStack
import random
from utils.lora import LoRANetwork, DEFAULT_TARGET_REPLACE, UNET_TARGET_REPLACE_MODULE_CONV
from transformers import logging
logging.set_verbosity_warning()
from diffusers import logging
logging.set_verbosity_error()

parser = argparse.ArgumentParser()
parser.add_argument("--lora_name", type=str)
parser.add_argument("--version", type=str)
parser.add_argument("--num_inference_steps", type=int)
parser.add_argument("--train_iteration", type=int)
parser.add_argument("--save_per_steps", type=int)
parser.add_argument("--device", type=str)
parser.add_argument("--output_dir", type=str)
parser.add_argument("--datasets_folder", type=str)
parser.add_argument("--dataset",type=str)
parser.add_argument("--source_folder",type=str)
parser.add_argument("--target_folder",type=str)
parser.add_argument("--cfg_eta",type=float)
parser.add_argument("--start_step", type=int)
parser.add_argument("--end_step", type=int)
parser.add_argument("--guidance_scale",type=float)
parser.add_argument("--pretrained_model_name_or_path",type=str)
parser.add_argument("--image_id", type=int)

args = parser.parse_args()
num_inference_steps = args.num_inference_steps
lora_name = args.lora_name

device = args.device

max_train_steps = args.train_iteration
save_per_steps = args.save_per_steps
output_dir = args.output_dir

dataset = args.dataset

source_folder = args.source_folder
target_folder = args.target_folder

cfg_eta = args.cfg_eta

start_step = args.start_step
end_step = args.end_step
guidance_scale = args.guidance_scale
image_id = args.image_id

import os
os.environ["CUDA_VISIBLE_DEVICES"]=device
device = f"cuda:{args.device}"

import torch

pretrained_model_name_or_path = args.pretrained_model_name_or_path
weight_dtype = torch.bfloat16


max_sequence_length = 512
height = width = 512 


# optimizer params
lr = 0.002


# lora params
alpha = 1
rank = 16
# train_method = 'xattn'
train_method = 'full'

# training params
batchsize = 1

def flush():
    torch.cuda.empty_cache()
    gc.collect()
flush()


# Load the tokenizers
tokenizer_one = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer", torch_dtype=weight_dtype, device_map=device)
tokenizer_two = T5TokenizerFast.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer_2", torch_dtype=weight_dtype, device_map=device)

# Load scheduler and models
noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler",torch_dtype=weight_dtype, device_map=device)

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

tokenizers = [tokenizer_one, tokenizer_two]
text_encoders = [text_encoder_one, text_encoder_two]


prompt_embeds_arr = []
pooled_prompt_embeds_arr = []
text_ids_arr = []


prompt_embeds, pooled_prompt_embeds, text_ids = compute_text_embeddings(
    [""], text_encoders, tokenizers, max_sequence_length
)
target_prompt_embeds = prompt_embeds[0]
target_pooled_prompt_embeds = pooled_prompt_embeds[0]

target_text_ids = text_ids[0]


params = []
modules = DEFAULT_TARGET_REPLACE
modules += UNET_TARGET_REPLACE_MODULE_CONV

network_content = LoRANetwork(
    transformer,
    rank=rank,
    multiplier=0.0,
    alpha=alpha,
    train_method=train_method,
).to(device, dtype=weight_dtype)

params.extend(network_content.prepare_optimizer_params())
    
optimizer = AdamW(params, lr=lr)
optimizer.zero_grad()

criteria = torch.nn.MSELoss()

pipe = FluxPipeline(noise_scheduler,
    vae,
    text_encoder_one,
    tokenizer_one,
    text_encoder_two,
    tokenizer_two,
    transformer,
)
pipe.set_progress_bar_config(disable=False)


lr_warmup_steps = 200
lr_num_cycles = 1
lr_power = 1.0
lr_scheduler = 'constant' 
#Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]
lr_scheduler = get_scheduler(
    lr_scheduler,
    optimizer=optimizer,
    num_warmup_steps=lr_warmup_steps,
    num_training_steps=max_train_steps,
    num_cycles=lr_num_cycles,
    power=lr_power,
)
    
progress_bar = tqdm(
    range(0, max_train_steps),
    desc="Steps",
)

losses = {}

# log_dir = "./logs_test"  # 日志保存的目录
# writer = SummaryWriter(log_dir=log_dir)

l1_loss = torch.nn.L1Loss()
criteria = torch.nn.MSELoss()

generator = None

vae_scale_factor = 2 ** (len(vae.config.block_out_channels))
height_model = 2 * (int(height) // pipe.vae_scale_factor)
width_model = 2 * (int(width) // pipe.vae_scale_factor)


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


ims_sources = os.listdir(f'{source_folder}/')
ims_sources = [im_ for im_ in ims_sources if '.png' in im_ or '.jpg' in im_ or '.jpeg' in im_ or '.webp' in im_]
ims_sources = sorted(ims_sources)
image_id_name = [x for x in ims_sources if x.startswith(f"{image_id}.")][0]

for epoch in range(max_train_steps):
    timestep_to_infer = random.randint(start_step, end_step)
    timesteps = torch.tensor([noise_scheduler.timesteps[timestep_to_infer]]).to(device=device)

    with torch.no_grad():
        img1 = Image.open(f'{source_folder}/{image_id_name}').resize((512,512))

        seed = random.randint(0,2*15)

        init_latents, denoised_latents, noise = pipe.add_noise_rect(img1, timestep_to_infer, torch.Generator().manual_seed(seed))
        
        denoised_latents = denoised_latents.to(device, dtype=weight_dtype)
        noise = noise.to(device, dtype=weight_dtype)

        denoised_latents_unpack = FluxPipeline._unpack_latents(
            denoised_latents,
            height=height,
            width=width,
            vae_scale_factor=vae_scale_factor,
        )

        init_latents = FluxPipeline._unpack_latents(
            init_latents,
            height=height,
            width=width,
            vae_scale_factor=vae_scale_factor,
        )


    if epoch == 0:
        model_input = FluxPipeline._unpack_latents(
            denoised_latents,
            height=height,
            width=width,
            vae_scale_factor=vae_scale_factor,
        )

    noise = FluxPipeline._unpack_latents(
        noise,
        height=int(model_input.shape[2] * vae_scale_factor / 2),
        width=int(model_input.shape[3] * vae_scale_factor / 2),
        vae_scale_factor=vae_scale_factor,
    )
    
    latent_image_ids = FluxPipeline._prepare_latent_image_ids(
        model_input.shape[0], model_input.shape[2], 
        model_input.shape[3], 
        device, weight_dtype,
    )

    # handle guidance
    if transformer.config.guidance_embeds:
        guidance = torch.tensor([guidance_scale], device=device)
        guidance = guidance.expand(denoised_latents.shape[0])
    else:
        guidance = None

    network_content.set_lora_slider(1)
    with ExitStack() as stack:
        stack.enter_context(network_content)

        model_noise_pred_neg = transformer(
            hidden_states=denoised_latents,
           timestep=timesteps / 1000,
            guidance=guidance,
            pooled_projections=target_pooled_prompt_embeds,
            encoder_hidden_states=target_prompt_embeds,
            txt_ids=target_text_ids,
            img_ids=latent_image_ids,
            return_dict=False,
        )[0]


    model_pred_neg = FluxPipeline._unpack_latents(
        model_noise_pred_neg,
        height=int(model_input.shape[2] * vae_scale_factor / 2),
        width=int(model_input.shape[3] * vae_scale_factor / 2),
        vae_scale_factor=vae_scale_factor,
    )

    loss1 = criteria(model_pred_neg, noise - init_latents)
    loss1 = loss1.mean()
    loss1.backward()


    logs = {"loss1": loss1.item(),"lr": lr_scheduler.get_last_lr()[0]}
    
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
    
    progress_bar.update(1)
    progress_bar.set_postfix(**logs)


    if (epoch) % save_per_steps == 0 and epoch != 0:
        # Save the trained LoRA model
        save_path = f'{output_dir}'
        os.makedirs(save_path, exist_ok=True)

        print("Saving...")
        output_weight = f"{save_path}/{dataset}_{image_id}_{num_inference_steps}inf_{epoch}epoch.pt"
        network_content.save_weights(
            output_weight,
            dtype=weight_dtype,
        )
    
print('Training Done')

# Save the trained LoRA model
save_path = f'{output_dir}'
os.makedirs(save_path, exist_ok=True)

print("Saving...")
network_content.save_weights(
    f"{save_path}/{dataset}_{image_id}_{num_inference_steps}inf_{epoch+1}epoch.pt",
    dtype=weight_dtype,
)

flush()
print("Done.")