import sys
sys.path.append('..')
import os
import yaml
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--c", type=str, help="config name")

args = parser.parse_args()

config_name = args.c

config_file_name = f'config/{config_name}.yaml'

with open(f'{config_file_name}', 'r') as file:
    config = yaml.safe_load(file)

config_name = config['config_name']
train_iteration = config['train_iteration']
num_inference_steps = config['num_inference_steps']

datasets = config['datasets']

python_name = config['python_name']
device = config['device']

start_step = config['start_step']
end_step = config['end_step']
cfg_eta = config['cfg_eta']

guidance_scale = config['guidance_scale']

train_count = config['train_count']
train_start = config['train_start']
train_method=config['train_method']
output_folder = config['output_folder']

pretrained_model_name_or_path = config['pretrained_model_name_or_path']

skips = config['skips']

use_prompts_file = config['use_prompts_file']
use_random_seed = config['use_random_seed']

output_folder_gen = config['output_folder_gen']

prompts = config['prompt']

prompts_file = config['prompts_file']

count_gen = config['count_gen']
scales = config['scales']

use_trained_lora_path = config['use_trained_lora_path']
trained_lora_path = config['trained_lora_path']

noise_scales = config['noise_scales']

start_seed_gen = config['start_seed_gen']
semantic_lora_path = config['semantic_lora_path']
semantic_cfg_scale = config['semantic_cfg_scale']
image_ids = config['image_ids']

os_rs = True
for index in range(len(datasets)):
    dataset = datasets[index]

    if index >= len(prompts_file):
        prompts_file_gen = prompts_file[-1]
    else:
        prompts_file_gen = prompts_file[index]
    if index >= len(prompts):
        prompt = prompts[-1]
    else:
        prompt = prompts[index]

    for i in range(train_start, train_start + train_count):
        for image_id in image_ids:
            lora_name=f"{dataset}_{image_id}_v{python_name}_{config_name}_v{i}"
            lora_path=f"{output_folder}/{dataset}/{lora_name}/{dataset}_{image_id}_{num_inference_steps}inf_{train_iteration}epoch.pt"

            output_path=f"{output_folder_gen}/{dataset}/{lora_name}"

            if use_trained_lora_path:
                lora_path = f'{trained_lora_path}'

            for skip in skips:
                command = f'PYTHONPATH=".." python generate_real_image.py ' 
                command += f' --lora_path="{lora_path}" '
                command += f' --dataset="{dataset}"'
                command += f' --lora_name="{lora_name}"'
                command += f' --prompts_file="prompts/{prompts_file_gen}"'
                command += f' --num_inference_steps={num_inference_steps}'
                command += f' --prompt="{prompt}"'
                command += f' --skip={skip}'
                command += f' --start_seed={start_seed_gen}'
                command += f' --count={count_gen}'
                command += f' --scales="{scales}"'
                command += f' --output_path="{output_path}"'
                command += f' --guidance_scale={guidance_scale}'
                command += f' --device={device}'
                command += f' --epoch={train_iteration}'
                command += f' --use_prompts_file={use_prompts_file}'
                command += f' --use_random_seed={use_random_seed}'
                command += f' --pretrained_model_name_or_path={pretrained_model_name_or_path}'
                command += f' --semantic_lora_path="{semantic_lora_path}"'
                command += f' --semantic_cfg_scale={semantic_cfg_scale}'

                rs = os.system(command)
                if rs != 0:
                    os_rs = False
                    break
            
            if not os_rs:
                break
        if not os_rs:
                break
    if not os_rs:
                break

