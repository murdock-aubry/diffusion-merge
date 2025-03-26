import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from diffusers import DiffusionPipeline, DDIMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from accelerate import Accelerator
import numpy as np
from PIL import Image
import os
import gc

def clear_gpu_memory():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

def release_memory(*args):
    """Helper function to delete variables and clear memory"""
    for var in args:
        if var is not None:
            del var
    clear_gpu_memory()

# Configuration
model_id = "CompVis/stable-diffusion-v1-4"
dataset_name = "reach-vb/pokemon-blip-captions"
output_dir = "./pokemon_finetuned_model"
num_train_epochs = 5
batch_size = 5  # Keep small batch size for memory constraints
learning_rate = 1e-7
resolution = 256
gradient_accumulation_steps = 4  # Increased to help with small batch size
max_grad_norm = 1.0

# Initialize Accelerator with better memory settings
accelerator = Accelerator(
    gradient_accumulation_steps=gradient_accumulation_steps,
    mixed_precision='fp16'  
)

# Load dataset
dataset = load_dataset(dataset_name, split="train")

# Load models with memory-efficient settings
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(
    model_id, 
    subfolder="text_encoder",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

# Load pipeline with memory optimizations
pipe = DiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    use_safetensors=True,
    low_cpu_mem_usage=True
)
vae = pipe.vae
unet = pipe.unet
unet.enable_gradient_checkpointing()  # Critical for memory savings
scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

# Optimized preprocessing
def preprocess(examples):
    # Tokenize captions
    inputs = tokenizer(
        examples["text"],
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids
    
    # Process images with memory efficiency
    images = []
    for img_data in examples["image"]:
        img_array = np.array(img_data, dtype=np.uint8)
        if img_array.ndim == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        
        # Process directly to tensor with reduced precision
        image = torch.from_numpy(img_array).float() / 127.5 - 1.0  # [-1, 1] range
        image = image.permute(2, 0, 1)  # CHW format
        image = torch.nn.functional.interpolate(
            image.unsqueeze(0),
            size=(resolution, resolution),
            mode="bilinear"
        ).squeeze(0)
        images.append(image.half())  # Store as float16
    
    pixel_values = torch.stack(images)
    return {"pixel_values": pixel_values, "input_ids": input_ids}

dataset.set_transform(preprocess)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model preparation
vae.eval()
text_encoder.eval()
unet.train()

# Freeze models that shouldn't be trained
vae.requires_grad_(False)
text_encoder.requires_grad_(False)

# Prepare with accelerator
optimizer = torch.optim.Adam(
    unet.parameters(),
    lr=1e-6,
    betas=(0.9, 0.999),  # More conservative than AdamW's defaults
    eps=1e-8,
    weight_decay=0
)

unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)

vae = vae.to(accelerator.device)
text_encoder = text_encoder.to(accelerator.device)



for epoch in range(num_train_epochs):
    for step, batch in enumerate(dataloader):
        with accelerator.accumulate(unet):
            # Forward pass
            with torch.no_grad():
                latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
                latents = latents * 0.18215
                text_embeddings = text_encoder(batch["input_ids"])[0]
            
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps,
                (latents.shape[0],),
                device=latents.device
            ).long()
            
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)
            
            # Simple fp32 forward pass
            noise_pred = unet(noisy_latents, timesteps, 
                            encoder_hidden_states=text_embeddings).sample
            # noise_pred = 0.1 * noise_pred + 0.9 * noisy_latents 

            loss = torch.nn.functional.mse_loss(
                noise_pred.float(), 
                noise.float(), 
                reduction="none"
            ).mean([1, 2, 3]).mean() 
            loss = loss.clamp(max=1e4)
            
            # Backward pass
            accelerator.backward(loss)

            if any(torch.isnan(p.grad).any() for p in unet.parameters() if p.grad is not None):
                optimizer.zero_grad()
                continue
            
            # Gradient clipping and NaN checks
            if accelerator.sync_gradients:
                # Check for NaN/inf in loss
                if torch.isnan(loss) or torch.isinf(loss):
                    optimizer.zero_grad()
                    print("Skipping step due to NaN/inf loss")
                    continue
                    
                # Gradient clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    unet.parameters(), 
                    max_grad_norm
                )
                
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    optimizer.zero_grad()
                    print("Skipping step due to NaN/inf gradients")
                    continue
            
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            # Logging
            
            print(f"Epoch: {epoch}, step: {step}, loss: {loss.item():.4f}, "
                     f"grad_norm: {grad_norm.item() if 'grad_norm' in locals() else 0:.4f}")

# Save the model with memory considerations
accelerator.wait_for_everyone()
if accelerator.is_main_process:
    unet = accelerator.unwrap_model(unet)
    pipe = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,  # Disable safety checker to save memory
        feature_extractor=None,  # Disable feature extractor
    )
    pipe.save_pretrained(output_dir, safe_serialization=True)
    print(f"Model saved to {output_dir}")