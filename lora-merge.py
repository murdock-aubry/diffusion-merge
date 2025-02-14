from diffusers import DiffusionPipeline
import torch
from utils import *

# Define model values

base_model = "stabilityai/stable-diffusion-xl-base-1.0"

model1 = "ostris/ikea-instructions-lora-sdxl"
model1_weight = "ikea_instructions_xl_v1_5.safetensors"
model1_name = "ikea"
model1_contribution = 0.7


model2 = "lordjia/by-feng-zikai"
model2_weight = "fengzikai_v1.0_XL.safetensors"
model2_name = "feng"
model2_contribution = 0.8


# Load pretrained stable diffusion model
pipeline = DiffusionPipeline.from_pretrained(
    base_model, 
    torch_dtype=torch.float16
).to("cuda")


# Load LoRAs using the pipeline.
pipeline.load_lora_weights(model1, weight_name=model1_weight, adapter_name=model1_name)
pipeline.load_lora_weights(model2, weight_name=model2_weight, adapter_name=model2_name)

# merge adapters with 
manual_set_adapters(pipeline, [model1_name, model2_name], [model1_contribution, model2_contribution])


# Run inference
generator = torch.manual_seed(0)
prompt = "A guy named James who is attractive and capable of performing advanced mathematics"
image = pipeline(prompt, generator=generator, cross_attention_kwargs={"scale": 1.0}).images[0]
image.save("outputs/manual_james_mathematics.png")