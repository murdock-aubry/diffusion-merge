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
        # Initialize parameters with zeros
        for param in merged_unet.parameters():
            param.data.zero_()
        
        # Add weighted parameters from each model
        for unet, weight in zip(unets, weights):
            for merged_param, model_param in zip(merged_unet.parameters(), unet.parameters()):
                merged_param.data += weight * model_param.data
    
    return merged_unet