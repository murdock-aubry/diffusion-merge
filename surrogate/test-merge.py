import torch
from diffusers import StableDiffusionPipeline
from surrogate.models import MergedDiffusionModel
from vae.vae import TimeConditionedVAE
from mlp.models import TimeConditionedMLP


if __name__ == "__main__":

    latent_dim = 819

    # MLP Params
    time_embed_dim = 64
    hidden_dims = [512, 256, 512]

    # Load trained VAEs
    vae_a_ckpt = torch.load("/w/383/murdock/models/vae/sd1.4/dim819_epoch100_time_conditioned.pth")
    vae_a = TimeConditionedVAE(latent_dim=latent_dim)
    vae_a.load_state_dict(vae_a_ckpt["model_state_dict"])

    

    vae_b_ckpt = torch.load("/w/383/murdock/models/vae/sd1.4-dogtuned/dim819_epoch100_time_conditioned.pth")
    vae_b = TimeConditionedVAE(latent_dim=latent_dim)
    vae_b.load_state_dict(vae_b_ckpt["model_state_dict"])

    # Load MLPs
    mlp_a_to_b_ckpt = torch.load("/w/383/murdock/models/mlp/sd1.4_to_sd1.4-dogtuned_dim819.pt")
    mlp_b_to_a_ckpt = torch.load("/w/383/murdock/models/mlp/sd1.4-dogtuned_to_sd1.4_dim819.pt")

    mlp_a_to_b = TimeConditionedMLP(
            latent_dim,
            hidden_dims = hidden_dims,
            time_embed_dim = time_embed_dim
        )

    mlp_b_to_a = TimeConditionedMLP(
            latent_dim,
            hidden_dims = hidden_dims,
            time_embed_dim = time_embed_dim
        )

    mlp_a_to_b.load_state_dict(mlp_a_to_b_ckpt)
    mlp_b_to_a.load_state_dict(mlp_b_to_a_ckpt)
    
    # surrogate_model = SurrogateDiffusionModel(
    #     "CompVis/stable-diffusion-v1-4",
    #     "frknayk/dreambooth_training",
    #     vae_a,
    #     vae_b,
    #     mlp_a_to_b,
    #     mlp_b_to_a
    # )

    merged_model = MergedDiffusionModel(
        "CompVis/stable-diffusion-v1-4",
        "frknayk/dreambooth_training",
        vae_a,
        vae_b,
        mlp_a_to_b,
        mlp_b_to_a
    )
    

    print("Surrogate model constructed")

    images = merged_model(
        prompt="a photo of an astronaut riding a horse on mars",
        guidance_scale=7.5,
        num_inference_steps=50
    )
    
    images[0].save("merged_diffusion_output.png")