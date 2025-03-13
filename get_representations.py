import torch
import os
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline
from datasets import load_dataset
import json 
import random
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
import io
from utils import *


def extract_hidden_reps(pipe, prompt, num_steps=50, model_name="empty_name"):
    """
    Extract UNet hidden representations (inputs and outputs) for each timestep.
    
    Args:
        pipe (StableDiffusionXLPipeline): The pre-loaded SDXL pipeline.
        prompt (str): The text prompt for inference.
        num_steps (int): Number of inference steps (default: 50).
        model_name (str): Name of the model being used (default: "empty_name").
    
    Returns:
        tuple: (inputs tensor, outputs tensor) where each tensor has shape [num_steps, hidden_dimension]
    """
    import torch
    
    # Lists to store tensors for each timestep
    inputs_list = []
    outputs_list = []
    
    # Hook to capture UNet input and output
    def unet_hook_fn(module, input, output):
        # Store input latent
        inputs_list.append(input[0][0].detach().cpu())  # x_t, note input[0][0] == input[0][1] for all layers
        
        # Store output noise prediction
        outputs_list.append(output[0][0].detach().cpu())  # epsilon
    
    # Register the hook
    hook_handle = pipe.unet.register_forward_hook(unet_hook_fn)
    
    # Set a fixed seed for reproducibility
    generator = torch.Generator(device="cuda").manual_seed(42)
    
    # Run inference
    with torch.no_grad():
        _ = pipe(
            prompt=prompt,
            num_inference_steps=num_steps,
            output_type="latent",
            generator=generator,
            num_images_per_prompt=1  # Batch size of 1
        )
    
    # Remove the hook
    hook_handle.remove()
    
    # Then concatenate along the first dimension
    inputs_tensor = torch.stack(inputs_list, dim=0)
    outputs_tensor = torch.stack(outputs_list, dim=0)
    
    return inputs_tensor, outputs_tensor


def get_prompts(source="fixed", num_samples=5):
    """Retrieve prompts from a dataset or use fixed ones."""

    if source == "fixed":
        return [
        "a corgi"
    ]
    else:
        # random.seed(42)
        dataset = load_dataset(source, split="train").shuffle(seed=42)
        
        if num_samples == -1:
            return [dataset[i]["Prompt"] for i in range(len(dataset))]
        else:
            return [dataset[i]["Prompt"] for i in range(num_samples)]

if __name__ == "__main__":
    # Set seed for reproducibility
    random.seed(42)

    config = load_json_file("config.json")

    # model_name_read = "sd1.4-dogtuned"
    # model_name_read = "sd1.4-cocotuned"
    model_name_read = "sd1.4"
    model_name = config["models"][model_name_read]
    
    data_link = "nateraw/parti-prompts"
    num_samples = -1
    dataset = get_prompts(source=data_link, num_samples=num_samples)

    # File to save data
    output_file = f"/w/383/murdock/hidden_reps/{model_name_read}/representations.pt"

    # Load existing data if it exists, otherwise initialize
    if os.path.exists(output_file):
        data = torch.load(output_file)
    else:
        data = {
            "prompts": [],
            "inputs": [],
            "outputs": []
        }

    # Get existing prompts as a set for fast lookup
    existing_prompts = set(data["prompts"])


    # Filter out prompts that already exist
    unique_dataset = [prompt for prompt in dataset if prompt not in existing_prompts]

    # If no new prompts, exit early
    if not unique_dataset:
        print("No new unique prompts to process. All prompts already exist in the file.")
    else:
        # Load the pipeline only if there’s work to do
        pipe = DiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16
        ).to("cuda")

        # Process only unique prompts and append to .pt file
        for iprompt, prompt in enumerate(unique_dataset):
            # Extract hidden representations
            inputs, outputs = extract_hidden_reps(pipe, prompt, num_steps = 50, model_name=model_name)
            
            # Append new data
            data["prompts"].append(prompt)
            data["inputs"].append(inputs)
            data["outputs"].append(outputs)

            # Save updated data back to the .pt file after each iteration
            torch.save(data, output_file)

            # Optional: Print progress
            print(f"Saved prompt {iprompt + 1}/{len(unique_dataset)}: {prompt}")

        print(f"All new data saved to {output_file}. Total unique prompts processed: {len(unique_dataset)}")