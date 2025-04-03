from diffusers import DiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_cosine_schedule_with_warmup
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from torchvision import transforms
from tqdm.auto import tqdm
import os
import gc
import numpy as np
import random
import argparse
import io
from PIL import Image
import base64


parser = argparse.ArgumentParser(description="Specify dataset name for finetuning.")
parser.add_argument("--data_shard", type=str, required=True, help="Name of the dataset to use for finetuning.")
parser.add_argument("--data_path", type=str, required=True, help="Path of the dataset to use for finetuning.")
args = parser.parse_args()



DATASET_NAME = args.data_shard
DATASET_PATH = args.data_path



# Configuration

print("Initializing hyperparameters", flush = True)
MODEL_NAME = "CompVis/stable-diffusion-v1-4"
OUTPUT_DIR = f"/projects/dynamics/diffusion-tmp/finetunes/{DATASET_NAME}"
BATCH_SIZE = 5  # Reduce batch size
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 1e-5
NUM_EPOCHS = 5
MIXED_PRECISION = "no"  # Use mixed precision

# Enhanced memory optimization configuration
torch.backends.cudnn.benchmark = True  # Optimize GPU performance
torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for matrix multiplications
torch.backends.cudnn.allow_tf32 = True

# Set the appropriate dtype
torch_dtype = torch.float16 if MIXED_PRECISION == "fp16" else torch.float32

class MemoryEfficientPromptImageDataset(Dataset):
    def __init__(self, dataset_name, split="train"):

        # loading pre-split shards based on category name

        self.dataset = load_dataset(DATASET_PATH, split="train")
        if len(self.dataset) > 1000:
            indices = list(range(len(self.dataset)))
            shuffled_indices = random.Random(42).sample(indices, 1000)
            self.dataset = self.dataset.select(shuffled_indices)
            
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),  # Consistent image size
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]

        image = item["image"]
        image = self.transform(image)

        if DATASET_NAME in ["ghibli"]:
            prompt = item["caption"] 
        else:
            prompt = item["text"] 

        return {"prompt": prompt, "pixel_values": image}

def setup_model_for_training(model_name, torch_dtype):
    # Load model with aggressive memory saving
    pipe = DiffusionPipeline.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        safety_checker=None,  # Disable safety checker
        variant="fp16" if torch_dtype == torch.float16 else None
    )
    
    # Move to GPU with memory optimization
    pipe = pipe.to("cuda")
    
    # Freeze most components
    pipe.text_encoder.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    
    # More aggressive memory optimization
    pipe.enable_attention_slicing("max")  # Maximum memory savings
    pipe.enable_vae_slicing()
    pipe.unet.enable_gradient_checkpointing()
    
    return pipe

def train_model(pipe, train_dataloader, num_epochs):
    # Optimizer with memory-efficient settings
    optimizer = torch.optim.AdamW(
        pipe.unet.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-6
    )
    
    # Learning rate scheduler
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=len(train_dataloader) * NUM_EPOCHS
    )
    
    # Training loop with enhanced memory management
    for epoch in range(num_epochs):
        pipe.unet.train()
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}")
        
        for step, batch in enumerate(train_dataloader):
            # Use context manager for mixed precision
            with torch.autocast("cuda", dtype=torch_dtype):
                # Explicitly manage tensor devices and precision
                images = batch["pixel_values"].to("cuda", dtype=torch_dtype)
                prompts = batch["prompt"]
                
                # Encode latents with minimal memory
                with torch.no_grad():
                    latents = pipe.vae.encode(images).latent_dist.sample() * 0.18215
                    del images
                
                # Tokenize and encode text
                input_ids = pipe.tokenizer(
                    prompts,
                    padding="max_length",
                    max_length=pipe.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                ).input_ids.to("cuda")
                
                with torch.no_grad():
                    encoder_hidden_states = pipe.text_encoder(input_ids)[0]
                    del input_ids
                
                # Prepare noise and timesteps
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, pipe.scheduler.num_train_timesteps, 
                    (latents.shape[0],), 
                    device=latents.device
                ).long()
                
                # Add noise
                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
                del latents
                
                # Predict noise
                noise_pred = pipe.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    return_dict=False
                )[0]
                
                # Compute loss
                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                loss = loss / GRADIENT_ACCUMULATION_STEPS
            
            # Backward pass with gradient scaling
            loss.backward()

            print(f"epoch: {epoch}, step: {step}, loss: {loss}", flush = True)
            
            # Gradient clipping and optimization step
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": loss.item() * GRADIENT_ACCUMULATION_STEPS})
            
            # Explicit memory cleanup
            del loss, noise_pred, noisy_latents, timesteps, encoder_hidden_states
            torch.cuda.empty_cache()
        
    # Save checkpoint
    pipe.save_pretrained(os.path.join(OUTPUT_DIR, f"epoch-{epoch}"))
    torch.cuda.empty_cache()

def main():
    # Prepare dataset

    print("Loading training dataset", flush = True)

    train_dataset = MemoryEfficientPromptImageDataset(DATASET_NAME)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        pin_memory=True,  # Improve data transfer to GPU
        num_workers=2  # Parallel data loading
    )

    print("Loading model pipeline", flush = True)

    # Setup and train model
    pipe = setup_model_for_training(MODEL_NAME, torch_dtype)
    train_model(pipe, train_dataloader, NUM_EPOCHS)
    
    print("Training complete!", flush = True)

if __name__ == "__main__":
    main()