import torch
from diffusers import DiffusionPipeline, UNet2DConditionModel
import os
from PIL import Image
from tqdm import tqdm
import numpy as np

# Create output directory if it doesn't exist
output_dir = "denoising_steps"
os.makedirs(output_dir, exist_ok=True)

# Path to the original model
base_model_path = "CompVis/stable-diffusion-v1-4"

custom_unet_name = "sd1.4_sd1.4-cocotuned_thresh0.0"
# custom_unet_name = "blank"

# Path to your saved UNet
custom_unet_path = f"/w/383/murdock/models/unets/zipit/{custom_unet_name}"


# 1. Load the original pipeline with safety checker disabled
pipeline = DiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    safety_checker=None
)

device = "cuda" if torch.cuda.is_available() else "cpu"
pipeline = pipeline.to(device)



# 2. Load  custom UNet
try:
    custom_unet = UNet2DConditionModel.from_pretrained(
        custom_unet_path,
        torch_dtype=torch.float16
    )

    pipeline.unet = custom_unet

except Exception as e:
    print(f"Error loading custom UNet, using original: {e}")


# 3. Move to GPU
pipeline = pipeline.to("cuda")
unet = pipeline.unet

prompt = "A hyper-intelligent robot alien looking over a sea of planets, stars, and galaxies in space."


# 3. Run inference with torch.no_grad()
with torch.no_grad():
    # Generate the image
    image = pipeline(prompt).images[0]
    
    # Save the image if needed
    from datetime import datetime
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    image.save(f"outputs/{custom_unet_name}_{current_time}.png")


quit()



prompt = "a photo of a cat"
with torch.no_grad():
    # Generate initial noise
    latents = torch.randn(
        (1, pipeline.unet.config.in_channels, 64, 64),  # Adjust size if needed
        dtype=torch.float16,
        device="cuda"
    )
    print("Initial latents - NaN:", torch.isnan(latents).any().item(), "min:", latents.min().item(), "max:", latents.max().item())

    # Denoising loop (simplified)
    pipeline.scheduler.set_timesteps(50)
    for t in pipeline.scheduler.timesteps:
        text_input = pipeline.tokenizer(prompt, padding="max_length", max_length=pipeline.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_embeddings = pipeline.text_encoder(text_input["input_ids"].to("cuda"))[0]
        unet_output = pipeline.unet(latents, t, encoder_hidden_states=text_embeddings).sample
        print(f"UNet output - NaN:", torch.isnan(unet_output).any().item(), "min:", unet_output.min().item(), "max:", unet_output.max().item())
        
        # If NaN, re-run with layer-by-layer inspection
        if torch.isnan(unet_output).any():
            print("Debugging UNet layers...")
            x = latents
            for name, module in pipeline.unet.named_modules():
                if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, torch.nn.GroupNorm)):
                    x_prev = x.clone()
                    x = module(x)
                    if torch.isnan(x).any():
                        print(f"NaN introduced in {name}")
                        break
        latents = pipeline.scheduler.step(unet_output, t, latents).prev_sample

    # Decode with VAE
    latents = latents / pipeline.vae.config.scaling_factor  # Adjust for VAE scaling
    image = pipeline.vae.decode(latents).sample
    print("VAE output - NaN:", torch.isnan(image).any().item(), "min:", image.min().item(), "max:", image.max().item())

    # Post-process
    image = (image / 2 + 0.5).clamp(0, 1)  # Normalize from [-1, 1] to [0, 1]
    print("Normalized image - NaN:", torch.isnan(image).any().item(), "min:", image.min().item(), "max:", image.max().item())
    image = image.cpu().numpy()
    image = (image * 255).round().clip(0, 255).astype("uint8")
    print("Final image - min:", image.min(), "max:", image.max())
    # Save the final image

    image = np.transpose(image, (0, 2, 3, 1))  # Assuming (B, C, H, W) -> (B, H, W, C)
    image = image[0]  # Take the first image if batch dimension exists

    # Save the image
    image_pil = Image.fromarray(image)
    image_pil.save("output.png")




quit()



# 4. Define a custom callback function to save intermediate steps
def save_latents_callback(step, timestep, latents):
    # Convert latents to image
    with torch.no_grad():
        latents = 1 / 0.18215 * latents
        image = pipeline.vae.decode(latents).sample
        
        # Process the image 
        image = (image / 2 + 0.5).clamp(0, 1)
        
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()

        
        images = (image * 255).round().astype("uint8")
        pil_image = Image.fromarray(images[0])
        
        # Save the image
        filename = os.path.join(output_dir, f"step_{step:04d}.png")
        pil_image.save(filename)
        
        # Print some diagnostic info
        pixel_min = images[0].min()
        pixel_max = images[0].max()
        pixel_mean = images[0].mean()
        print(f"Step {step}/{num_steps}: min={pixel_min}, max={pixel_max}, mean={pixel_mean:.2f}")
        
        return latents

# 5. Prompt and generation parameters
prompt = "a photo of a sunset over mountains"
num_steps = 50  # More steps to see the progression clearly
guidance_scale = 7.5

# 6. Generate with callback
print(f"Generating image with {num_steps} steps, saving intermediates to {output_dir}...")
try:
    # Set a fixed seed for reproducibility
    generator = torch.manual_seed(42)
    
    image = pipeline(
        prompt,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        callback=save_latents_callback,
        callback_steps=1,  # Save every step
        generator=generator
    ).images[0]
    
    # Save the final image
    final_path = os.path.join(output_dir, "final_image.png")
    image.save(final_path)
    print(f"Final image saved to {final_path}")
    
except Exception as e:
    print(f"Error during generation: {e}")
    
    # Try to get more detailed error information
    import traceback
    traceback.print_exc()

# 7. Create a diagnostic summary of saved images
try:
    print("\nAnalyzing saved steps:")
    step_files = [f for f in os.listdir(output_dir) if f.startswith("step_")]
    step_files.sort()
    
    if not step_files:
        print("No step files were saved, something went wrong during generation.")
    else:
        print(f"Successfully saved {len(step_files)} intermediate steps.")
        
        # Check if all images are black
        all_black = True
        for file in step_files:
            img = Image.open(os.path.join(output_dir, file))
            img_array = torch.tensor(list(img.getdata())).reshape(img.size[1], img.size[0], -1)
            if img_array.max() > 0:
                all_black = False
                break
                
        if all_black:
            print("WARNING: All saved steps appear to be black images.")
            print("This suggests a fundamental issue with the UNet or VAE processing.")
        else:
            print("At least some of the saved steps contain non-black pixels.")
except Exception as e:
    print(f"Error analyzing steps: {e}")