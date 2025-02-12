from diffusers import DiffusionPipeline
import torch

base_model = "stabilityai/stable-diffusion-xl-base-1.0"

lora1 = "ostris/ikea-instructions-lora-sdxl"
lora1_weights = "ikea_instructions_xl_v1_5.safetensors"
lora1_name = "ikea"

lora2 = "lordjia/by-feng-zikai"
lora2_weights = "fengzikai_v1.0_XL.safetensors"
lora2_name = "feng"

pipeline = DiffusionPipeline.from_pretrained(base_model, torch_dtype=torch.float16).to("cuda")
pipeline.load_lora_weights(lora1, weight_name=lora1_weights, adapter_name=lora1_name)
pipeline.load_lora_weights(lora2, weight_name=lora2_weights, adapter_name=lora2_name)

pipeline.set_adapters([lora1_name, lora2_name], adapter_weights=[0.2, 0.8])


generator = torch.manual_seed(0)
prompt = "A guy named James who is attractive and capable of performing advanced mathematics"


image = pipeline(prompt, generator=generator, cross_attention_kwargs={"scale": 1.0}).images[0]


image.save("outputs/james_mathematics_2_8.png")