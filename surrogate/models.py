import torch
from diffusers import StableDiffusionPipeline

class SurrogateDiffusionModel:

    #             Unet 1
    #       X_t ---------> X_{t-1}
    #        |              |
    #    VAE |              | VAE
    #    MLP |              | MLP
    #    VAE |              | VAE
    #        |              |
    #       X_t ---------> X_{t-1}
    #             Unet 2
    

    def __init__(
        self,
        model1,
        model2,
        vae_a,
        vae_b,
        mapping_mlp_a_to_b,
        mapping_mlp_b_to_a,
        num_diffusion_steps=50,
        device="cuda"
    ):
        # Load models - base model A and finetuned model B
        # Load model A and extract UNet
        self.model_a = StableDiffusionPipeline.from_pretrained(model1).to(device)
        self.unet_a = self.model_a.unet
        self.decoder_a = self.model_a.vae.decoder
        self.numpy_to_pil = self.model_a.numpy_to_pil
        self.vae_scale = self.model_a.vae.config.scaling_factor
        del self.model_a
        torch.cuda.empty_cache()


        # Load model B and extract UNet
        self.model_b = StableDiffusionPipeline.from_pretrained(model2).to(device)
        self.unet_b = self.model_b.unet

        # Use model B's components for generation
        self.text_encoder = self.model_b.text_encoder
        self.tokenizer = self.model_b.tokenizer
        self.scheduler = self.model_b.scheduler
        self.vae = self.model_b.vae

        del self.model_b
        torch.cuda.empty_cache()
        
        # Load compression VAEs and mapping networks
        self.vae_a = vae_a.to(device)
        self.vae_a.eval()

        self.vae_b = vae_b.to(device)
        self.vae_b.eval()

        self.mapping_mlp_a_to_b = mapping_mlp_a_to_b.to(device)
        self.mapping_mlp_b_to_a = mapping_mlp_b_to_a.to(device)

        self.num_diffusion_steps = num_diffusion_steps
        
        self.device = device

    def _normalize_time(self, t):
        """Convert timestep to normalized time in range [0, 1]"""
        return t / self.num_diffusion_steps
        
        
    def _process_latent(self, latent, t, encoder_hidden_states):
        
        """Process the latent through the surrogate diffusion process"""

        t = t.to(self.device)
        batch_size = latent.shape[0]
        t = t.repeat(batch_size)
        t_emb = self.vae_a.encode_time(t)

        mu_a, logvar_a = self.vae_a.encode(latent, t_emb)
        compressed_a = self.vae_a.reparameterize(mu_a, logvar_a)
        
        # 2. Map from A's compressed space to B's compressed space
        compressed_b = self.mapping_mlp_a_to_b(compressed_a, t)

        # 3. Decode using VAE B
        mapped_latent = self.vae_b.decode(compressed_b, t_emb)
        
        # 4. Pass through UNet B to get predicted noise
        noise_pred = self.unet_b(mapped_latent, t, encoder_hidden_states=encoder_hidden_states).sample

        # 5. Compress the predicted noise using VAE B
        mu_b_noise, log_b_noise = self.vae_b.encode(noise_pred, t_emb)
        compressed_noise_b = self.vae_b.reparameterize(mu_b_noise, log_b_noise)

        # 6. Map back to A's compressed space
        compressed_noise_a = self.mapping_mlp_b_to_a(compressed_noise_b, t)
        
        # 7. Decode using VAE A
        final_noise_pred = self.vae_a.decode(compressed_noise_a, t_emb)
        
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
        self.scheduler.set_timesteps(num_inference_steps-1)
    

        time_indices = torch.arange(num_inference_steps).float()
        # Normalize time indices to [0, 1]
        normalized_time = time_indices / (num_inference_steps - 1)
        
        
        latents = latents * self.scheduler.init_noise_sigma

        # Denoising loop
        for i, t in enumerate(self.scheduler.timesteps):

            t_norm = normalized_time[i]


            # If we're doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if negative_prompt is not None else latents
            
            # Scale for input to the models
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # Get noise prediction through our surrogate process
            noise_pred = self._process_latent(
                latent_model_input, 
                t_norm,
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
            image = self.decoder_a(latents / self.vae_scale)
        
        # Convert to PIL Image
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).round().astype("uint8")
        image = [self.numpy_to_pil(img)[0] for img in image]
        
        return image
    



class MergedDiffusionModel:

    #             Unet 1
    #       X_t ---------> X_{t-1}
    #        |              |
    #    VAE |              | VAE
    #    MLP |              | MLP
    #    VAE |              | VAE
    #        |              |
    #       X_t ---------> X_{t-1}
    #             Unet 2
    

    def __init__(
        self,
        model1,
        model2,
        vae_a,
        vae_b,
        mapping_mlp_a_to_b,
        mapping_mlp_b_to_a,
        num_diffusion_steps=50,
        alpha=0.0,
        device="cuda"
    ):
        # Load models - base model A and finetuned model B
        # Load model A and extract UNet
        self.model_a = StableDiffusionPipeline.from_pretrained(model1).to(device)
        self.unet_a = self.model_a.unet
        del self.model_a
        torch.cuda.empty_cache()


        # Load model B and extract UNet
        self.model_b = StableDiffusionPipeline.from_pretrained(model2).to(device)
        self.unet_b = self.model_b.unet

        # Use model B's components for generation
        self.decoder_b = self.model_b.vae.decoder
        self.numpy_to_pil = self.model_b.numpy_to_pil
        self.vae_scale = self.model_b.vae.config.scaling_factor
        self.text_encoder = self.model_b.text_encoder
        self.tokenizer = self.model_b.tokenizer
        self.scheduler = self.model_b.scheduler
        self.vae = self.model_b.vae

        del self.model_b
        torch.cuda.empty_cache()
        
        # Load compression VAEs and mapping networks
        self.vae_a = vae_a.to(device)
        self.vae_a.eval()

        self.vae_b = vae_b.to(device)
        self.vae_b.eval()

        self.mapping_mlp_a_to_b = mapping_mlp_a_to_b.to(device)
        self.mapping_mlp_b_to_a = mapping_mlp_b_to_a.to(device)

        self.num_diffusion_steps = num_diffusion_steps
        
        self.device = device

        self.alpha = alpha

    def _normalize_time(self, t):
        """Convert timestep to normalized time in range [0, 1]"""
        return t / self.num_diffusion_steps
        
        
    def _process_latent(self, latent, t, encoder_hidden_states):
        
        """Process the latent through the surrogate diffusion process"""

        """Assumption: Latents live in some ambient representation space interpretable by both models"""

        # Run inference with B on the original latent image, obtain plain noise predicted directly from B
        noise_pred_original = self.unet_b(latent, t, encoder_hidden_states=encoder_hidden_states).sample
        
        
        t = t.to(self.device)
        batch_size = latent.shape[0]
        t = t.repeat(batch_size)
        t_emb = self.vae_a.encode_time(t)


        # Encode the input latents using VAE-A, which pre-assumes model A's latent structure
        mu_a, logvar_a = self.vae_a.encode(latent, t_emb)
        compressed_a = self.vae_a.reparameterize(mu_a, logvar_a)

        # Map encoded latents of A to the encoded space of model B
        compressed_b = self.mapping_mlp_a_to_b(compressed_a, t)

        # decode the mapped latents using the compressed latents
        mapped_latents = self.vae_b.decode(compressed_b, t_emb)

        # Run inference on the mapped latents injected with A's structure using the weights of B.
        noise_pred_mapped = self.unet_b(mapped_latents, t, encoder_hidden_states=encoder_hidden_states).sample

        # Encode the mapped predicted noise back to the space of A
        mu_b_noise, log_b_noise = self.vae_b.encode(noise_pred_mapped, t_emb)
        compressed_noise_b = self.vae_b.reparameterize(mu_b_noise, log_b_noise)

        # Map back to A's compressed space
        compressed_noise_a = self.mapping_mlp_b_to_a(compressed_noise_b, t)
        
        # Decode using VAE A
        noise_pred_mapped = self.vae_a.decode(compressed_noise_a, t_emb)

        # Interpolate between original noise and the mapped noise
        final_noise_pred = (1.0 - self.alpha) * noise_pred_original + self.alpha * noise_pred_mapped

        
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
            (1, self.unet_b.config.in_channels, height // 8, width // 8),
            device=self.device,
        )
        
        # Set up scheduler
        self.scheduler.set_timesteps(num_inference_steps-1)
    

        time_indices = torch.arange(num_inference_steps).float()
        # Normalize time indices to [0, 1]
        normalized_time = time_indices / (num_inference_steps - 1)
        
        
        latents = latents * self.scheduler.init_noise_sigma

        # Denoising loop
        for i, t in enumerate(self.scheduler.timesteps):

            t_norm = normalized_time[i]

            # If we're doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if negative_prompt is not None else latents
            
            # Scale for input to the models
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # Get noise prediction through our surrogate process
            noise_pred = self._process_latent(
                latent_model_input, 
                t_norm,
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
            image = self.decoder_b(latents / self.vae_scale)
        
        # Convert to PIL Image
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).round().astype("uint8")
        image = [self.numpy_to_pil(img)[0] for img in image]
        
        return image