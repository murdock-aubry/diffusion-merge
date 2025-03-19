import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.loaders import AttnProcsLayers
from diffusers.models.attention_processor import LoRAAttnProcessor
from tqdm import tqdm
import argparse
from PIL import Image
from torchvision import transforms
from pathlib import Path

# Define argument parser
parser = argparse.ArgumentParser(description="Train lightweight LoRA for Stable Diffusion model distillation")
parser.add_argument("--base_model", type=str, default="CompVis/stable-diffusion-v1-4", help="Base model path")
parser.add_argument("--dataset", type=str, required=True, help="Path to prompt-image dataset directory")
parser.add_argument("--output_dir", type=str, default="lora_output", help="Output directory")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size (keep small for memory)")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
parser.add_argument("--max_steps", type=int, default=1000, help="Maximum training steps")
parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X steps")
parser.add_argument("--lora_rank", type=int, default=4, help="LoRA rank (smaller = more lightweight)")
parser.add_argument("--validation_prompt", type=str, default="a photo of a cat", help="Prompt for validation")
args = parser.parse_args()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Load the base model (with half precision if using GPU)
print(f"Loading base model: {args.base_model}")
precision = torch.float16 if torch.cuda.is_available() else torch.float32
base_model = StableDiffusionPipeline.from_pretrained(args.base_model, torch_dtype=precision)

# Extract components we need
tokenizer = base_model.tokenizer
text_encoder = base_model.text_encoder.to(device)
vae = base_model.vae.to(device)
unet = base_model.unet.to(device)
noise_scheduler = DDPMScheduler.from_pretrained(args.base_model, subfolder="scheduler")

# Define simple dataset for prompt-image pairs
class SimplePromptImageDataset(Dataset):
    def __init__(self, dataset_path, tokenizer, image_size=512):
        self.tokenizer = tokenizer
        self.image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        
        # Find all image files and their corresponding prompt files
        self.dataset_path = Path(dataset_path)
        self.items = []
        
        for img_path in self.dataset_path.glob("*.png"):
            txt_path = img_path.with_suffix(".txt")
            if txt_path.exists():
                with open(txt_path, "r") as f:
                    prompt = f.read().strip()
                self.items.append((img_path, prompt))
        
        print(f"Loaded dataset with {len(self.items)} prompt-image pairs")
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        img_path, prompt = self.items[idx]
        
        # Load and process image
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.image_transform(image)
        
        # Process prompt
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "image": image_tensor,
            "input_ids": tokens.input_ids[0],
            "prompt": prompt
        }

# Apply LoRA to UNet's attention layers (minimalistic approach)
lora_attn_procs = {}
for name in unet.attn_processors.keys():
    # Only apply to cross-attention layers to keep it minimal
    if name.endswith("attn2.processor"):
        cross_attention_dim = unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        
        # Create LoRA attention processor with small rank
        lora_attn_procs[name] = LoRAAttnProcessor(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            rank=args.lora_rank,
        )

# Apply our minimal LoRA processors and get trainable params
unet.set_attn_processor(lora_attn_procs)
lora_layers = AttnProcsLayers(unet.attn_processors)
lora_layers.to(device)

# Freeze everything except LoRA parameters
unet.requires_grad_(False)
vae.requires_grad_(False)
text_encoder.requires_grad_(False)
lora_layers.requires_grad_(True)

# Simple optimizer
optimizer = torch.optim.AdamW(lora_layers.parameters(), lr=args.lr)

# Create dataset and dataloader
dataset = SimplePromptImageDataset(args.dataset, tokenizer)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

# Helper function to get latents from images
def encode_images_to_latents(vae, images):
    with torch.no_grad():
        latents = vae.encode(images.to(vae.dtype)).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
    return latents

# Training loop
print("Starting training...")
global_step = 0
progress_bar = tqdm(range(args.max_steps))

while global_step < args.max_steps:
    for batch in dataloader:
        if global_step >= args.max_steps:
            break
            
        # Get images and text embeddings
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        
        # Convert images to latent space
        with torch.no_grad():
            # Generate latents
            latents = encode_images_to_latents(vae, images)
            
            # Get text embeddings
            text_embeddings = text_encoder(input_ids)[0]
        
        # Add noise to latents
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Predict noise with UNet
        noise_pred = unet(noisy_latents, timesteps, text_embeddings).sample
        
        # Simple loss - MSE between predicted and actual noise
        loss = F.mse_loss(noise_pred, noise)
        
        # Update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update progress
        progress_bar.update(1)
        progress_bar.set_description(f"Loss: {loss.item():.4f}")
        global_step += 1
        
        # Save checkpoint
        if global_step % args.save_steps == 0 or global_step == args.max_steps:
            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            os.makedirs(save_path, exist_ok=True)
            
            # Save LoRA weights
            unet.save_attn_procs(save_path)
            
            # Generate a validation sample
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.base_model,
                torch_dtype=precision
            )
            pipeline.unet.load_attn_procs(save_path)
            pipeline = pipeline.to(device)
            
            # Generate validation image
            image = pipeline(args.validation_prompt, num_inference_steps=25).images[0]
            image.save(os.path.join(save_path, "validation.png"))
            
            print(f"Saved checkpoint at step {global_step}")

# Save final model
final_path = os.path.join(args.output_dir, "final_lora")
os.makedirs(final_path, exist_ok=True)
unet.save_attn_procs(final_path)

print(f"Training complete! Final LoRA weights saved to {final_path}")
print("To use the trained model:")
print(f"  1. Load base model: pipeline = StableDiffusionPipeline.from_pretrained('{args.base_model}')")
print(f"  2. Load LoRA weights: pipeline.unet.load_attn_procs('{final_path}')")
print("  3. Generate images: pipeline(prompt).images[0]")