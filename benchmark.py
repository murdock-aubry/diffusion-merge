import torch
import numpy as np
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline
from datasets import load_dataset
from torchmetrics.functional.multimodal.clip_score import clip_score
from functools import partial
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception_score import InceptionScore
import torchvision.transforms as T
from torchvision.metrics import structural_similarity_index_measure
from torchmetrics.image.lpips import LearnedPerceptualImagePatchSimilarity


def load_model(model_ckpt="stabilityai/stable-diffusion-xl-base-1.0"):
    """Load Stable Diffusion pipeline with the specified model checkpoint."""
    return StableDiffusionXLPipeline.from_pretrained(model_ckpt, torch_dtype=torch.float16).to("cuda")


def get_prompts(source="fixed", num_samples=5):
    """Retrieve prompts from a dataset or use fixed ones."""
    if source == "dataset":
        dataset = load_dataset("nateraw/parti-prompts", split="train").shuffle()
        return [dataset[i]["Prompt"] for i in range(num_samples)]
    
    return [
        "a corgi",
        "a hot air balloon with a yin-yang symbol, with the moon visible in the daytime sky",
        "a car with no windows",
        "a cube made of porcupine",
        "The saying 'BE EXCELLENT TO EACH OTHER' written on a red brick wall with a graffiti image of a green alien wearing a tuxedo. A yellow fire hydrant is on a sidewalk in the foreground.",
    ]


def generate_images(pipeline, prompts, num_images_per_prompt=1):
    """Generate images using the diffusion model."""
    results = pipeline(prompts, num_images_per_prompt=num_images_per_prompt, output_type="np").images
    return np.array(results)  # Ensure it's a NumPy array for processing


def calculate_clip_score(images, prompts, model_name="openai/clip-vit-base-patch16"):
    """Compute CLIP score for generated images."""
    clip_fn = partial(clip_score, model_name_or_path=model_name)
    images_int = (images * 255).astype("uint8")
    return round(float(clip_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()), 4)


def calculate_ssim(real_images, generated_images):
    transform = T.Resize((256, 256))  # Ensure size consistency
    real_resized = transform(real_images)
    generated_resized = transform(generated_images)
    return structural_similarity_index_measure(real_resized, generated_resized).item()

def calculate_fid(real_images, generated_images):
    fid = FrechetInceptionDistance(feature=2048).to("cuda")
    fid.update(real_images, real=True)
    fid.update(generated_images, real=False)
    return fid.compute().item()

def calculate_inception_score(generated_images):
    is_metric = InceptionScore(feature=2048).to("cuda")
    is_metric.update(generated_images)
    return is_metric.compute().item()

def calculate_lpips(real_images, generated_images):
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="alex").to("cuda")
    return lpips_metric(real_images, generated_images).item()

if __name__ == "__main__":
    torch.manual_seed(42)
    model_ckpt = "stabilityai/stable-diffusion-xl-base-1.0"
    
    pipeline = load_model(model_ckpt)
    prompts = get_prompts(source="fixed")  # Change to 'dataset' for dynamic prompts
    images = generate_images(pipeline, prompts, num_images_per_prompt=1)
    clip_score_value = calculate_clip_score(images, prompts)
    
    print(f"Model: {model_ckpt}\nCLIP Score: {clip_score_value}")
