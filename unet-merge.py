import torch
from diffusers import DiffusionPipeline
from utils import *

unet1 = "stabilityai/stable-diffusion-xl-base-1.0"
weight1 = 0.7
# Load pretrained stable diffusion model
pipeline1 = DiffusionPipeline.from_pretrained(
    unet1, 
    torch_dtype=torch.float16
).to("cuda")

# Extract U-Net model
unet1 = pipeline1.unet



unet2 = "cyberagent/opencole-stable-diffusion-xl-base-1.0-finetune"
weight2 = 0.3
pipeline2 = DiffusionPipeline.from_pretrained(
    unet2, 
    torch_dtype=torch.float16
).to("cuda")
unet2 = pipeline2.unet 


with torch.no_grad():
    for (name1, param1), (name2, param2) in zip(unet1.named_parameters(), unet2.named_parameters()):
        print(f"{name1 == name2}, {name1}, {name2}")  


quit()

# Merge U-Net with itself
merged_unet = merge_unets(unet1, unet2, weight1=0.5, weight2=0.5)
pipeline1.unet = merged_unet

# Run inference
generator = torch.manual_seed(0)
prompt = "A futuristic cityscape with flying cars"
image = pipeline1(prompt, generator=generator, cross_attention_kwargs={"scale": 1.0}).images[0]
image.save("outputs/merged_unet_test.png")