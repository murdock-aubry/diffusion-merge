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

# Configuration
MODEL_NAME = "CompVis/stable-diffusion-v1-4"
DATASET_NAME = "reach-vb/pokemon-blip-captions"
OUTPUT_DIR = "./fine-tuned-model"
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 5e-7  # Reduced from 1e-5
NUM_EPOCHS = 3
MIXED_PRECISION = "fp16"  # Try "no" if this still fails

# Set the appropriate dtype
torch_dtype = torch.float16 if MIXED_PRECISION == "fp16" else torch.float32

# Load dataset with validation
class PromptImageDataset(Dataset):
    def __init__(self, dataset_name, split="train"):
        self.dataset = load_dataset(dataset_name, split=split)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"].convert("RGB")
        prompt = item["text"]
        
        if not isinstance(image, torch.Tensor):
            image = self.transform(image)
            
        # Validate image data
        if torch.isnan(image).any() or torch.isinf(image).any():
            raise ValueError(f"Invalid image data at index {idx}")
            
        return {"prompt": prompt, "pixel_values": image}

# Prepare dataset
train_dataset = PromptImageDataset(DATASET_NAME)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)

# Load model with safety checks
pipe = DiffusionPipeline.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch_dtype,
    safety_checker=None,  # Disable safety checker to save memory
)
pipe = pipe.to("cuda")

# Freeze components
pipe.text_encoder.requires_grad_(False)
pipe.vae.requires_grad_(False)

# More aggressive memory optimization
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
pipe.unet.enable_gradient_checkpointing()

# Optimizer with more conservative settings
optimizer = torch.optim.AdamW(
    pipe.unet.parameters(),
    lr=LEARNING_RATE,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=1e-6
)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=100,
    num_training_steps=len(train_dataloader) * NUM_EPOCHS,
)

def clear_cuda_cache():
    torch.cuda.empty_cache()
    gc.collect()

def validate_tensor(tensor, name=""):
    if torch.isnan(tensor).any():
        raise ValueError(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        raise ValueError(f"Inf detected in {name}")
    return tensor

global_step = 0
for epoch in range(NUM_EPOCHS):
    pipe.unet.train()
    progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}")
    
    for step, batch in enumerate(train_dataloader):
        try:
            clear_cuda_cache()
            
            # Get batch with validation
            images = validate_tensor(
                batch["pixel_values"].to("cuda", non_blocking=True).to(torch_dtype),
                "input images"
            )
            prompts = batch["prompt"]
            
            # Latent encoding with validation
            with torch.no_grad():
                latents = pipe.vae.encode(images).latent_dist.sample() * 0.18215
                latents = validate_tensor(latents, "latents")
                del images
                clear_cuda_cache()
            
            # Text encoding with validation
            input_ids = pipe.tokenizer(
                prompts,
                padding="max_length",
                max_length=pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).input_ids.to("cuda", non_blocking=True)
            
            with torch.no_grad():
                encoder_hidden_states = pipe.text_encoder(input_ids)[0].to(torch_dtype)
                encoder_hidden_states = validate_tensor(encoder_hidden_states, "encoder_hidden_states")
                del input_ids
                clear_cuda_cache()
            
            # Noise and timesteps with validation
            noise = validate_tensor(
                torch.randn_like(latents).to(torch_dtype),
                "initial noise"
            )
            timesteps = torch.randint(
                0, pipe.scheduler.num_train_timesteps, (latents.shape[0],), device=latents.device
            ).long()
            
            # Add noise with validation
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
            noisy_latents = validate_tensor(noisy_latents, "noisy_latents")
            del latents, noise
            clear_cuda_cache()
            
            # Forward pass with gradient checkpointing
            with torch.autocast("cuda", dtype=torch_dtype):
                noise_pred = pipe.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    return_dict=False
                )[0]
                noise_pred = validate_tensor(noise_pred, "noise_pred")
            
            # Loss calculation with new target noise
            target_noise = validate_tensor(
                torch.randn_like(noise_pred),
                "target noise"
            )
            loss = torch.nn.functional.mse_loss(noise_pred, target_noise)
            loss = validate_tensor(loss, "loss") / GRADIENT_ACCUMULATION_STEPS
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(), 0.5)  # More aggressive clipping
            
            del noisy_latents, timesteps, encoder_hidden_states, noise_pred, target_noise
            clear_cuda_cache()
            
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            
            progress_bar.update(1)
            global_step += 1
            
            if global_step % 100 == 0:
                progress_bar.set_postfix({"loss": loss.item() * GRADIENT_ACCUMULATION_STEPS})

            print(f"Epoch: {epoch}, step: {step}, loss: {loss.item() * GRADIENT_ACCUMULATION_STEPS}")
            
            del loss
            clear_cuda_cache()
            
        except Exception as e:
            print(f"Error at step {step}: {str(e)}")
            # Reset gradients and clear memory
            optimizer.zero_grad(set_to_none=True)
            clear_cuda_cache()
            # Skip to next batch
            continue
    
    # Save checkpoint
    pipe.save_pretrained(os.path.join(OUTPUT_DIR, f"epoch-{epoch}"))
    clear_cuda_cache()

print("Training complete!")