import torch
import numpy as np
from diffusers import StableDiffusionXLPipeline
from datasets import load_dataset
from torchmetrics.functional.multimodal.clip_score import clip_score
from torch_fidelity import calculate_metrics
from torchmetrics.image.inception import InceptionScore
import clip
import ImageReward as RM
import torch.nn.functional as F
import json

def load_model(model_ckpt="stabilityai/stable-diffusion-xl-base-1.0"):
    """Load Stable Diffusion pipeline with the specified model checkpoint."""
    return StableDiffusionXLPipeline.from_pretrained(model_ckpt, torch_dtype=torch.float16).to("cuda")

def get_prompts(source="fixed", num_samples=5):
    """Retrieve prompts from a dataset or use fixed ones."""

    if source == "fixed":
        return [
        "a corgi",
        "a hot air balloon with a yin-yang symbol, with the moon visible in the daytime sky",
        "a car with no windows",
        "a cube made of porcupine",
        "The saying 'BE EXCELLENT TO EACH OTHER' written on a red brick wall with a graffiti image of a green alien wearing a tuxedo. A yellow fire hydrant is on a sidewalk in the foreground.",
    ]
    else:

        dataset = load_dataset(source, split="train").shuffle()
        if num_samples == -1:
            return [dataset[i]["Prompt"] for i in range(len(dataset))]
        else:
            return [dataset[i]["Prompt"] for i in range(num_samples)]
        
def get_prompts_local(source="fixed", num_samples=5):
    """Retrieve prompts from a dataset or use fixed ones."""

    if source == "fixed":
        return [
        "a corgi",
        "a hot air balloon with a yin-yang symbol, with the moon visible in the daytime sky",
        "a car with no windows",
        "a cube made of porcupine",
        "The saying 'BE EXCELLENT TO EACH OTHER' written on a red brick wall with a graffiti image of a green alien wearing a tuxedo. A yellow fire hydrant is on a sidewalk in the foreground.",
    ]
    else:

        dataset = load_dataset('parquet', data_files=source, split = "train").shuffle()

        if num_samples == -1:
            return [dataset[i]["item"]["caption"] for i in range(len(dataset))]
        else:
            return [dataset[i]["item"]["caption"] for i in range(num_samples)]
            
    
    

def generate_images(pipeline, prompts, num_images_per_prompt=1):
    """Generate images using the diffusion model."""
    results = pipeline(prompts, num_images_per_prompt=num_images_per_prompt, output_type="np").images
    return np.array(results)  # Ensure it's a NumPy array for processing

def calculate_blip_score(generated_images, prompts, model, processor):
    """Compute similarity score for generated images using BLIP."""
    
    
    # Ensure images are properly formatted
    if generated_images.max() > 1.0:
        generated_images = generated_images / 255.0

    # BLIP typically expects images in the range [0,1]
    # Resize images to BLIP's expected size (typically 224x224)
    generated_images = F.interpolate(generated_images, size=(224, 224), mode='bilinear', align_corners=False)

    # Get model device and move images to that device
    device = next(model.parameters()).device
    generated_images = generated_images.to(device)
    
    # Process inputs
    if isinstance(prompts, str):
        # Process image and text
        inputs = processor(images=generated_images, text=prompts, return_tensors="pt").to(device)
        
        with torch.no_grad():
            # Forward pass through BLIP
            outputs = model(**inputs)
            
            # Extract features for similarity calculation
            # For BLIP, we need to extract image and text embeddings
            # The exact API depends on which BLIP model you're using
            if hasattr(model, "get_image_features") and hasattr(model, "get_text_features"):
                # For BLIP models with explicit feature extractors
                image_features = model.get_image_features(generated_images)
                text_features = model.get_text_features(inputs.input_ids)
                
                # Normalize features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity
                similarity = torch.cosine_similarity(image_features, text_features)
            else:
                # For BLIP models without explicit feature extractors,
                # use the ITM (Image-Text Matching) score
                itm_outputs = model(**inputs, output_attentions=False)
                itm_scores = itm_outputs.logits
                similarity = torch.nn.functional.softmax(itm_scores, dim=1)[:, 1]  # Probability of match

    return similarity


def calculate_clip_score(generated_images, prompts, model_name="openai/clip-vit-base-patch16"):
    """Compute CLIP score for generated images."""
    # Ensure images are properly formatted for CLIP
    if generated_images.max() > 1.0:
        generated_images = generated_images / 255.0
    
    return round(float(clip_score(generated_images, prompts, model_name_or_path=model_name).detach()), 4)

# def calculate_clip_score(generated_images, prompts, model, tokenizer):
#     """Compute CLIP score for generated images."""
    
#     # Ensure images are properly formatted for CLIP
#     if generated_images.max() > 1.0:
#         generated_images = generated_images / 255.0

#     # Resize images to CLIP's expected size
#     generated_images = F.interpolate(generated_images, size=(224, 224), mode='bilinear', align_corners=False)

#     # Get model device and move images to that device
#     device = next(model.parameters()).device
#     generated_images = generated_images.to(device)
    
#     # Process text input - handle string prompts
#     if isinstance(prompts, str):
#         # Create tokenizer if needed
#         # tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")  # adjust model name if needed
#         text_inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
        
#         # Run the clip core
#         with torch.no_grad():
#             image_features = model.get_image_features(generated_images)
#             text_features = model.get_text_features(
#                 input_ids=text_inputs.input_ids,
#                 attention_mask=text_inputs.attention_mask
#             )
#             similarity = torch.cosine_similarity(image_features, text_features)

#     return similarity


def calculate_ir_score(generated_images, prompts, model):

    rewards = model.score(prompts, generated_images)

    return rewards

def compute_and_save_metrics(generated_images, prompts, model_ckpt):
    """Compute and save metrics."""

    inception = InceptionScore()


    print("computing CLIP")
    clip_score_value = calculate_clip_score(generated_images, prompts)

    metrics = {
        "clip_score": clip_score_value,
        # "inception_score": inception_score_value,
        # "diversity_coverage": diversity_coverage_value
    }

    with open(f"metrics.json", "w") as file:  # Removed model_ckpt from filename
        json.dump(metrics, file)

if __name__ == "__main__":
    torch.manual_seed(42)

    model_ckpt = "stabilityai/stable-diffusion-xl-base-1.0"
    pipeline = load_model(model_ckpt)
    prompts = get_prompts(source="dataset")

    images = generate_images(pipeline, prompts, num_images_per_prompt=1)
    
    # Convert to torch.Tensor, rearrange channels, scale to [0, 255]
    generated_images = torch.from_numpy(images).permute(0, 3, 1, 2).type(torch.uint8)

    # inception = InceptionScore()
    # inception.update(generated_images)
    # score = inception.compute()
    # print(f"Inception Score: {score}")

    # quit()



    if generated_images.max() > 1.0:
        generated_images = generated_images / 255.0

    compute_and_save_metrics(generated_images, prompts, model_ckpt)