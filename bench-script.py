import torch
import numpy as np
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline 
from datasets import load_dataset
import json
from benchmark import *
from utils import *
import argparse
from PIL import Image
import shutil
import tempfile
import os 


torch.manual_seed(42)

parser = argparse.ArgumentParser(description="Process some arguments.")
parser.add_argument("--model_names", type=str, nargs='+', required=True, help="Names of the models.")
parser.add_argument("--weights", type=float, nargs='+', required=False, default=[1.0], help="Weights for the models as a list of floats.")
parser.add_argument("--num-prompts", type=int, required=False, default=5, help="Number of prompts to process in dataset.")
args = parser.parse_args()

model_names = args.model_names
weights = np.array(args.weights)
num_prompts = args.num_prompts

if len(model_names) != len(weights):
    raise ValueError("Number of model names must match number of weights.")
    
if len(model_names) == 1:
    weights = [1.0]

if np.sum(weights) != 1:
    raise ValueError("Sum of weights must be 1.")

# Open the configuration file
with open('config.json', 'r') as file:
    config = json.load(file)

# Open existing metrics
with open('metrics.json', 'r') as file:
    metrics = json.load(file)


datasets = config["datasets"]
models = config["models"]

metric_names = ["clip"]


num_images_per_prompt = 1
num_prompts = 20

unets = []

model_name_merged = ""

# Load all models
if len(model_names) > 1:
    for imodel, model_name in enumerate(model_names):
        model_ckpt = models[model_name]
        pipeline = StableDiffusionXLPipeline.from_pretrained(model_ckpt, torch_dtype=torch.float16).to("cuda")
        unet = pipeline.unet 
        model_name_merged += f"{model_name}_{weights[imodel]}_"
        unets.append(unet)

    model_name = model_name_merged[-1]

    merged_unet = merge_unets(unets, weights)
    pipeline.unet = merged_unet

else:
    model_name = model_names[0]
    model_ckpt = models[model_name]
    pipeline = StableDiffusionXLPipeline.from_pretrained(model_ckpt, torch_dtype=torch.float16).to("cuda")
    


clip_name = "openai/clip-vit-base-patch16"

if model_name not in metrics.keys():
    metrics[model_name] = {}

for data_name, data_link in datasets.items():

    print(f"test prompts from {data_name}")

    if data_name not in metrics[model_name].keys():
        metrics[model_name][data_name] = {}

    dataset = get_prompts(source = data_link, num_samples = num_prompts)
    
    clips = 0
    ir = 0

    for iprompt, prompt in enumerate(dataset):

        print(f"processing prompt {iprompt + 1} of {num_prompts}")

        temp_dir = tempfile.mkdtemp()
        temp_image_path = os.path.join(temp_dir, "temp_image.png")

        # Generate images
        images = pipeline(prompt, num_images_per_prompt=num_images_per_prompt, output_type="np").images

        # Ensure it's a single image, convert data type
        image = images[0]  # Take the first image if it's a batch
        image = (image * 255).astype(np.uint8)  # Convert float [0,1] to uint8 [0,255]

        # Convert to PIL Image and save
        image_pil = Image.fromarray(image)
        image_pil.save(temp_image_path)

        # Convert to PyTorch tensor (NCHW format for CLIP model)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).type(torch.uint8)

        # Compute scores
        clips += calculate_clip_score(image_tensor, prompt, model_name=clip_name) / num_prompts
        ir += calculate_ir_score(temp_image_path, prompt) / num_prompts

        # Clean up
        shutil.rmtree(temp_dir)

    metrics[model_name][data_name]["nprompts"] = num_prompts
    metrics[model_name][data_name]["clip"] = clips
    metrics[model_name][data_name]["ir"] = ir

# Save the updated metrics to the file
with open('metrics.json', 'w') as file:
    json.dump(metrics, file)

