import torch
import numpy as np
from diffusers import DiffusionPipeline, UNet2DConditionModel
from datasets import load_dataset
import json
from benchmark import *
from utils import *
import argparse
from PIL import Image
import shutil
import tempfile
import os 
import gc


torch.manual_seed(42)

# parser = argparse.ArgumentParser(description="Process some arguments.")
# parser.add_argument("--model_path", type=str, required=True, help="Path to model ckpt.")
# parser.add_argument("--num-prompts", type=int, required=False, default=5, help="Number of prompts to process in dataset.")
# args = parser.parse_args()

# model_name = args.model_name
# weight_path = args.weight_path
num_prompts = 20#args.num_prompts

model_path = "/w/383/murdock/models/unets/zipit/"
# model_name = "sd1.4_sd1.4-cocotuned_thresh0.7"
model_name = "blank"
unet_ckpt = model_path + model_name


base_model_path = "CompVis/stable-diffusion-v1-4"

pipeline = DiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    safety_checker=None
)

device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = pipeline.to(device)


custom_unet = UNet2DConditionModel.from_pretrained(
    unet_ckpt,
    torch_dtype=torch.float16
).to(device)

pipeline.unet = custom_unet


# Open existing metrics
with open('metrics.json', 'r') as file:
    metrics = json.load(file)

with open('../config.json', 'r') as file:
    config = json.load(file)
datasets = config["datasets"]


gc.collect()
torch.cuda.empty_cache()


clip_name = "openai/clip-vit-base-patch16"

if model_name not in metrics.keys():
    metrics[model_name] = {}


for data_name, data_link in datasets.items():

    print(f"test prompts from {data_name}")

    if data_name not in metrics[model_name].keys():
        metrics[model_name][data_name] = {}

    dataset = get_prompts(source = data_link, num_samples = num_prompts)

    num_prompts = len(dataset)
    
    clips = 0
    ir = 0

    for iprompt, prompt in enumerate(dataset):

        print(f"processing prompt {iprompt + 1} of {num_prompts}")

        temp_dir = tempfile.mkdtemp()
        temp_image_path = os.path.join(temp_dir, "temp_image.png")

        # Generate images
        images = pipeline(prompt, num_images_per_prompt=1, output_type="np").images

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

        gc.collect()
        torch.cuda.empty_cache()

    metrics[model_name][data_name]["nprompts"] = num_prompts
    metrics[model_name][data_name]["clip"] = clips
    metrics[model_name][data_name]["ir"] = ir

# Save the updated metrics to the file
with open('metrics.json', 'w') as file:
    json.dump(metrics, file)

