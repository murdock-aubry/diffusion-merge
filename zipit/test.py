import torch
from diffusers import DiffusionPipeline, UNet2DConditionModel
import os
from PIL import Image
from tqdm import tqdm
import numpy as np


# Path to the original model
base_model_path = "CompVis/stable-diffusion-v1-4"
# base_model_path = "/scratch/ssd004/scratch/murdock/diffusion-merge/finetune/finetunes/animals/epoch-2"
# base_model_path = "animals_body-parts_clothes_electronics_food_text_vehicles_thresh0.0"


# custom_unet_name = "all_thresh0.0"
# custom_unet_name = "blank"

shard = "vehicles"


# Path to your saved UNet
model_path = "/projects/dynamics/diffusion-tmp/finetunes"
model_name = "animals_body-parts_clothes_electronics_food_text_vehicles_thresh0.0"
# model_name = "animals_thresh0.7"
custom_unet_path = f"{model_path}/{model_name}"


# 1. Load the original pipeline with safety checker disabled
pipeline = DiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float32,
    safety_checker=None
)

device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = pipeline.to(device)



# 2. Load  custom UNet

custom_unet = UNet2DConditionModel.from_pretrained(
    custom_unet_path,
    torch_dtype=torch.float32
)

pipeline.unet = custom_unet



# 3. Move to GPU
pipeline = pipeline.to("cuda")
unet = pipeline.unet

prompt = "A white bread sandwich on plate filled with ham and lettuce."


# 3. Run inference with torch.no_grad()
with torch.no_grad():
    # Generate the image
    image = pipeline(prompt).images[0]
    
    # Save the image if needed
    from datetime import datetime
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    image.save(f"outputs/{model_name}_{current_time}.png")


quit()

