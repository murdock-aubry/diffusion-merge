import torch
from diffusers import DiffusionPipeline
from utils import *

# unet2 = "cyberagent/opencole-stable-diffusion-xl-base-1.0-finetune"
unet2 = "cagliostrolab/animagine-xl-4.0"
pipeline2 = DiffusionPipeline.from_pretrained(
    unet2, 
    torch_dtype=torch.float16
).to("cuda")
unet2 = pipeline2.unet 

nimages = 100

for i in range(nimages + 1):

    unet1 = "stabilityai/stable-diffusion-xl-base-1.0"
    pipeline1 = DiffusionPipeline.from_pretrained(
        unet1, 
        torch_dtype=torch.float16
    ).to("cuda")
    # Extract U-Net model
    unet1 = pipeline1.unet

    p = round(1 - i / nimages, 2)
    q = round(1 - p, 2)

    # print(p,q)

    # Merge U-Net with itself
    merged_unet = merge_unets([unet1, unet2], [p, q])
    pipeline1.unet = merged_unet

    # Run inference
    generator = torch.manual_seed(0)
    prompt = "A lone warrior with flowing silver hair and piercing blue eyes, clad in an intricately detailed black and gold armored coat. A futuristic katana crackling with blue energy rests in their hand. Their expression is calm yet determined as they stand in an open field, surrounded by swirling mist."
    
    image = pipeline1(prompt, generator=generator, cross_attention_kwargs={"scale": 1.0}).images[0]
    image.save(f"outputs/unets/continuous2/{p}_{q}.png")

    del unet1, pipeline1, merged_unet


