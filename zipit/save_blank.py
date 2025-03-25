import torch
from diffusers import DiffusionPipeline

path = "CompVis/stable-diffusion-v1-4"
pipeline = DiffusionPipeline.from_pretrained(
        path, 
        torch_dtype=torch.float16
    ).to("cuda")

# Save a copy of the entire UNet
pipeline.unet.save_pretrained("/w/383/murdock/models/unets/zipit/blank")