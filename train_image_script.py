import sys
sys.path.append('..')
import os
import yaml
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--c", type=str, help="config name")

args = parser.parse_args()

config_name = args.c

timestamp = time.time()
time_tuple = time.localtime(timestamp)
formatted_time = time.strftime("%Y%m%d%H%M%S", time_tuple)

config_name = f"config/{config_name}.yaml"

with open(config_name, 'r') as file:
    config = yaml.safe_load(file)

config_name = config['config_name']
datasets_folder = config['datasets_folder']
datasets = config['datasets']
source_folder = config['source_folder']
target_folder = config['target_folder']
output_folder = config['output_folder']

device = config['device']
python_name = config['python_name']
train_iteration=config['train_iteration']
save_per_steps = config['save_per_steps']
num_inference_steps=config['num_inference_steps']
start_step=config['start_step']
end_step=config['end_step']
cfg_eta=config['cfg_eta']
noise_scales = config['noise_scales']

guidance_scale=config['guidance_scale']

train_count=config['train_count']
train_start=config['train_start']
train_method=config['train_method']

pretrained_model_name_or_path = config['pretrained_model_name_or_path']

train_rs = True
for dataset in datasets:
    source = f"{datasets_folder}/{dataset}/{source_folder}"
    target = f"{datasets_folder}/{dataset}/{target_folder}"

    for i in range(train_start, train_start + train_count):
        for noise_scale in noise_scales:
            lora_name=f"{dataset}_v{python_name}_{cfg_eta:.1f}eta_{noise_scale:.1f}ns_{train_method}_{config_name}_v{i}"
            save_path=f"{output_folder}/{dataset}/{lora_name}"

            command = f'PYTHONPATH=".." python {python_name}.py ' 
            command += f'--lora_name="{lora_name}" '
            command += f'--num_inference_steps={num_inference_steps} '
            command += f'--pretrained_model_name_or_path={pretrained_model_name_or_path} ' 
            command += f'--train_iteration={train_iteration} '
            command += f'--save_per_steps={save_per_steps} '
            command += f'--device="{device}" '  
            command += f'--output_dir="{save_path}" ' 
            command += f'--start_step={start_step} ' 
            command += f'--end_step={end_step} ' 
            command += f'--datasets_folder="{datasets_folder}" '
            command += f'--dataset="{dataset}" '
            command += f'--source_folder="{source}" '
            command += f'--target_folder="{target}" '
            command += f'--guidance_scale={guidance_scale} ' 
            command += f'--cfg_eta={cfg_eta} ' 
            command += f'--noise_scale={noise_scale} ' 
            print(command)

            rs = os.system(command)
            if rs != 0:
                train_rs = False
                break

        if not train_rs:
            break
    if not train_rs:
            break
