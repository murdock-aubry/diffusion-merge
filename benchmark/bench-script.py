import torch
import numpy as np
from diffusers import DiffusionPipeline, UNet2DConditionModel
from datasets import load_dataset
import json
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from benchmark import *
from utils import *
import argparse
from PIL import Image
import shutil
import argparse
import tempfile
import os 
import gc
import ImageReward as RM


torch.manual_seed(42)

# parser = argparse.ArgumentParser(description="Process some arguments.")
# parser.add_argument("--model_path", type=str, required=True, help="Path to model ckpt.")
# parser.add_argument("--num-prompts", type=int, required=False, default=5, help="Number of prompts to process in dataset.")
# args = parser.parse_args()

# model_name = args.model_name
# weight_path = args.weight_path

num_prompts = -1#args.num_prompts


model_path = "CompVis/stable-diffusion-v1-4"
# model_path = "/scratch/ssd004/scratch/murdock/diffusion-merge/finetune/finetunes/pokemon/epoch-2"

model_name = "sd1.4-base"

pipeline = DiffusionPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    safety_checker=None
)

device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = pipeline.to(device)



specialist = "animals_text_thresh0.7_no_base"
parser = argparse.ArgumentParser(description="Specify dataset name for finetuning.")
parser.add_argument("--data_shard", type=str, required=True, help="Name of the dataset to use for finetuning.")
args = parser.parse_args()


# specialist = args.data_shard
model_name = specialist#"merged-0.0"
unet_ckpt = f"/projects/dynamics/diffusion-tmp/finetunes/{specialist}"
custom_unet = UNet2DConditionModel.from_pretrained(
    unet_ckpt,
    torch_dtype=torch.float16
).to(device)

# pipeline.unet = custom_unet


# Open existing metrics

# path = "/projects/dynamics/diffusion-tmp/finetunes/"

# with open(f'/scratch/ssd004/scratch/murdock/diffusion-merge/benchmark/results/{specialist}.json', 'r') as file:
#     metrics = json.load(file)

with open('../config.json', 'r') as file:
    config = json.load(file)


datasets = config["datasets"]


gc.collect()
torch.cuda.empty_cache()

# clip_name = "openai/clip-vit-base-patch16"
# clip_model = CLIPModel.from_pretrained(clip_name).to(device)
# clip_processor = CLIPProcessor.from_pretrained(clip_name)
# clip_model.eval()
# clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")  # adjust model name if needed


ir_model = RM.load("ImageReward-v1.0")



# print(model_name)
# print(metrics.keys())

metrics = {}


print(f"benchmarking {model_name}", flush = True)

for data_name, data_link in datasets.items():

    print(f"test prompts from {data_name}, {data_link}", flush = True)


    metrics[data_name] = {}

    # print(metrics[model_name].keys(), flush = True)
    # if data_name not in metrics[model_name].keys():

    # else:
    #     print("data already saved")
    #     continue

    
    data_link = "/projects/dynamics/diffusion-tmp/data/test/" + data_link


    dataset = get_prompts_local(source = data_link, num_samples = num_prompts)

    num_prompts = len(dataset)
    
    clips = 0
    ir = 0

    for iprompt, prompt in enumerate(dataset):

        print(f"processing prompt {iprompt + 1} of {num_prompts}", flush = True)

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

        # blip_score = calculate_blip_score(image_tensor, prompt, blip_model, blip_processor)
        clips += calculate_clip_score(image_tensor, prompt) / num_prompts
        ir += calculate_ir_score(temp_image_path, prompt, ir_model) / num_prompts

        # Clean up
        shutil.rmtree(temp_dir)

        gc.collect()
        torch.cuda.empty_cache()

    metrics[data_name]["nprompts"] = num_prompts
    metrics[data_name]["clip"] = clips
    metrics[data_name]["ir"] = ir

    del dataset
    torch.cuda.empty_cache()


    # Save the updated metrics to the file
    with open(f'/scratch/ssd004/scratch/murdock/diffusion-merge/benchmark/results/{model_name}.json', 'w') as file:
        json.dump(metrics, file)

