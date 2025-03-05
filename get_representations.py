import torch
import os
from diffusers import StableDiffusionXLPipeline

def extract_hidden_reps(pipe, prompt, num_steps=50, output_dir="hidden_reps", model_name = "empty_name"):
    """
    Extract UNet hidden representations (inputs and outputs) for each timestep.
    
    Args:
        pipe (StableDiffusionXLPipeline): The pre-loaded SDXL pipeline.
        prompt (str): The text prompt for inference.
        num_steps (int): Number of inference steps (default: 50).
        output_dir (str): Directory to save hidden representations (default: "hidden_reps").
    
    Returns:
        tuple: (list of input file paths, list of output file paths)
    """

    output_dir += f"/{model_name}"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Lists to store file paths (not tensors, to save memory)
    input_paths = []
    output_paths = []

    # Hook to capture UNet input and output
    def unet_hook_fn(module, input, output):
        timestep = len(input_paths)
        input_path = f"{output_dir}/input_t{timestep}.pt"
        output_path = f"{output_dir}/output_t{timestep}.pt"
        torch.save(input[0].detach().cpu(), input_path)    # x_t
        torch.save(output[0].detach().cpu(), output_path)  # epsilon
        input_paths.append(input_path)
        output_paths.append(output_path)

    # Register the hook
    hook_handle = pipe.unet.register_forward_hook(unet_hook_fn)

    # Set a fixed seed for reproducibility
    generator = torch.Generator(device="cuda").manual_seed(42)

    # Run inference
    with torch.no_grad():
        _ = pipe(
            prompt=prompt,
            num_inference_steps=num_steps,
            output_type="latent",
            generator=generator,
            num_images_per_prompt=1  # Batch size of 1
        )

    # Remove the hook
    hook_handle.remove()

    return input_paths, output_paths

if __name__ == "__main__":

    # model_name = "stabilityai/stable-diffusion-xl-base-1.0"
    model_name = "cyberagent/opencole-stable-diffusion-xl-base-1.0-finetune"
    
    # Load the pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_name,
        torch_dtype=torch.float16
    ).to("cuda")

    # Extract hidden representations
    prompt = "A futuristic cityscape at sunset"
    input_paths, output_paths = extract_hidden_reps(pipe, prompt, model_name = model_name)

    # Print results
    print(f"Captured {len(input_paths)} timesteps:")
    for i, (inp, out) in enumerate(zip(input_paths, output_paths)):
        # Load tensors temporarily to check shapes
        inp_tensor = torch.load(inp)
        out_tensor = torch.load(out)
        print(f"Timestep {i}: Input shape: {inp_tensor.shape}, Output shape: {out_tensor.shape}")