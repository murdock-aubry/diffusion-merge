import torch


def manual_set_adapters(pipeline, adapter_names, adapter_weights=None):
    if isinstance(adapter_names, str):
        adapter_names = [adapter_names]

    if adapter_weights is None:
        adapter_weights = [1.0] * len(adapter_names)
    elif not isinstance(adapter_weights, list):
        adapter_weights = [adapter_weights] * len(adapter_names)

    if len(adapter_names) != len(adapter_weights):
        raise ValueError("Number of adapter names must match number of weights.")

    # Iterate through model components that support LoRA (e.g., unet, text_encoder)
    for component in ["unet", "text_encoder"]:

        # If no component in loaded pipeline, pass
        if not hasattr(pipeline, component):
            continue
        
        # extract particular component
        model = getattr(pipeline, component)

        # Apply LoRA weights manually to each adapter's parameters
        for adapter_name, weight in zip(adapter_names, adapter_weights):
            for name, param in model.named_parameters():
                if f"lora_{adapter_name}" in name:  # LoRA layers are usually prefixed with "lora_{adapter}"
                    param.data *= weight  # Scale adapter weights




def merge_unets(unets, weights):
    """
    Merges multiple U-Net models by combining their weights according to given weights.
    
    Args:
        unets (list): List of U-Net models to merge
        weights (list): List of corresponding weights for each model
        
    Returns:
        The merged U-Net model
    """
    if len(unets) != len(weights):
        raise ValueError("Number of models must match number of weights.")
    
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError("Sum of weights must be 1.")
    
    if len(unets) == 0:
        raise ValueError("At least one model must be provided.")
    
    # Create a copy of the first model to store the merged result
    merged_unet = unets[0]
    
    with torch.no_grad():
        # Initialize parameters with 0.5 their original value
        for param in merged_unet.parameters():
            param.data *= weights[0]
        
        # Add weighted parameters from each model
        for unet, weight in zip(unets[1:], weights[1:]):

            for merged_param, model_param in zip(merged_unet.parameters(), unet.parameters()):
                merged_param.data += weight * model_param.data
        
        
    
    return merged_unet


def load_tensors_from_directory(directory):
    tensors = {}
    for filename in os.listdir(directory):
        if filename.endswith('.pt'):
            tensor_name = filename[:-3]  # Remove the '.pt' extension
            tensors[tensor_name] = torch.load(os.path.join(directory, filename))
    return tensors


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