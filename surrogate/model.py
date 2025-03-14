import torch
from diffusers import StableDiffusionPipeline

class SurrogateDiffusionModel:

    #             Unet 1
    #       X_1 ---------> X_2
    #        |              |
    #    VAE |              | VAE
    #    MLP |              | MLP
    #    VAE |              | VAE
    #        |              |
    #       X_1 ---------> X_2
    #             Unet 2
    

    def __init__(
        self,
        base_model_id="CompVis/stable-diffusion-v1-4",
        finetuned_model_id="path/to/finetuned-model",
        vae_a,
        vae_b,
        mapping_mlp_a_to_b,
        mapping_mlp_b_to_a,
        device="cuda"
    ):
        # Load models - base model A and finetuned model B
        self.model_a = StableDiffusionPipeline.from_pretrained(base_model_id).to(device)
        self.model_b = StableDiffusionPipeline.from_pretrained(finetuned_model_id).to(device)
        
        # Extract UNets
        self.unet_a = self.model_a.unet
        self.unet_b = self.model_b.unet
        
        # Load compression VAEs and mapping networks
        self.vae_a = vae_a.to(device)
        self.vae_b = vae_b.to(device)
        self.mapping_mlp_a_to_b = mapping_mlp_a_to_b.to(device)
        self.mapping_mlp_b_to_a = mapping_mlp_b_to_a.to(device)
        
        self.device = device
        
        # Use model B's components for generation
        self.text_encoder = self.model_b.text_encoder
        self.tokenizer = self.model_b.tokenizer
        self.scheduler = self.model_b.scheduler
        self.vae = self.model_b.vae
        
    def _process_latent(self, latent, t, encoder_hidden_states):
        

        """Process the latent through the surrogate diffusion process"""
        # 1. Compress the input latent using VAE A
        compressed_a = self.vae_a.encode(latent).latent_dist.sample()
        
        # 2. Map from A's compressed space to B's compressed space
        compressed_b = self.mapping_mlp_a_to_b(compressed_a)
        
        # 3. Decode using VAE B
        mapped_latent = self.vae_b.decode(compressed_b)
        
        # 4. Pass through UNet B to get predicted noise
        noise_pred = self.unet_b(mapped_latent, t, encoder_hidden_states=encoder_hidden_states).sample
        
        # 5. Compress the predicted noise using VAE B
        compressed_noise_b = self.vae_b.encode(noise_pred).latent_dist.sample()
        
        # 6. Map back to A's compressed space
        compressed_noise_a = self.mapping_mlp_b_to_a(compressed_noise_b)
        
        # 7. Decode using VAE A
        final_noise_pred = self.vae_a.decode(compressed_noise_a)
        
        return final_noise_pred
    
    @torch.no_grad()
    def __call__(
        self,
        prompt,
        height=512,
        width=512,
        num_inference_steps=50,
        guidance_scale=7.5,
        negative_prompt=None,
    ):
        # Text encoding (using model B's text encoder)
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_inputs.input_ids.to(self.device))[0]
        
        # Negative prompt handling
        if negative_prompt is not None:
            uncond_input = self.tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # Initial random latent (using model A's configuration to preserve its semantic space)
        latents = torch.randn(
            (1, self.unet_a.config.in_channels, height // 8, width // 8),
            device=self.device,
        )
        
        # Set up scheduler
        self.scheduler.set_timesteps(num_inference_steps)
        latents = latents * self.scheduler.init_noise_sigma
        
        # Denoising loop
        for i, t in enumerate(self.scheduler.timesteps):
            # If we're doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if negative_prompt is not None else latents
            
            # Scale for input to the models
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # Get noise prediction through our surrogate process
            noise_pred = self._process_latent(
                latent_model_input, 
                t,
                encoder_hidden_states=text_embeddings
            )
            
            # Perform guidance if needed
            if negative_prompt is not None:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Compute previous image and set latents
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        # Decode the final latents with model A's VAE to stay in A's semantic space
        with torch.no_grad():
            image = self.model_a.vae.decode(latents / self.model_a.vae.config.scaling_factor).sample
        
        # Convert to PIL Image
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).round().astype("uint8")
        image = [self.model_a.numpy_to_pil(img)[0] for img in image]
        
        return image