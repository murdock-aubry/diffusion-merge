import torch
from diffusers import StableDiffusionPipeline

if __name__ == "__main__":
    # Load your trained VAEs and MLPs
    vae_a = torch.load("/w/383/murdock/models/vae/sd1.4/dim819_epoch100_time_conditioned.pth")
    vae_b = torch.load("/w/383/murdock/models/vae/sd1.4-dogtuned/dim819_epoch100_time_conditioned.pth")
    mapping_mlp_a_to_b = torch.load("/w/383/murdock/models/mlp/sd1.4_to_sd1.4-dogtuned_dim819.pt")
    mapping_mlp_b_to_a = torch.load("/w/383/murdock/models/mlp/sd1.4-dogtuned_to_sd1.4_dim819.pt")
    
    surrogate_model = SurrogateDiffusionModel(
        base_model_id="CompVis/stable-diffusion-v1-4",
        finetuned_model_id="frknayk/dreambooth_training",
        vae_a=vae_a,
        vae_b=vae_b,
        mapping_mlp_a_to_b=mapping_mlp_a_to_b,
        mapping_mlp_b_to_a=mapping_mlp_b_to_a
    )
    
    images = surrogate_model(
        prompt="a photo of an astronaut riding a horse on mars",
        guidance_scale=7.5,
        num_inference_steps=50
    )
    
    images[0].save("surrogate_diffusion_output.png")