import torch
from diffusers import DiffusionPipeline, UNet2DConditionModel
import os
from PIL import Image
from tqdm import tqdm
import numpy as np

# Create output directory if it doesn't exist
output_dir = "denoising_steps"
os.makedirs(output_dir, exist_ok=True)

# Path to the original model
base_model_path = "CompVis/stable-diffusion-v1-4"

custom_unet_name = "sd1.4_sd1.4-cocotuned_thresh0.0"
# custom_unet_name = "blank"

# Path to your saved UNet
custom_unet_path = f"/w/383/murdock/models/unets/zipit/{custom_unet_name}"


# 1. Load the original pipeline with safety checker disabled
pipeline = DiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    safety_checker=None
)

device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = pipeline.to(device)



# 2. Load  custom UNet
try:
    custom_unet = UNet2DConditionModel.from_pretrained(
        custom_unet_path,
        torch_dtype=torch.float16
    )

    pipeline.unet = custom_unet

except Exception as e:
    print(f"Error loading custom UNet, using original: {e}")


# 3. Move to GPU
pipeline = pipeline.to("cuda")
unet = pipeline.unet

prompt = "A hyper-intelligent robot alien looking over a sea of planets, stars, and galaxies in space."


# 3. Run inference with torch.no_grad()
with torch.no_grad():
    # Generate the image
    image = pipeline(prompt).images[0]
    
    # Save the image if needed
    from datetime import datetime
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    image.save(f"outputs/{custom_unet_name}_{current_time}.png")


quit()

