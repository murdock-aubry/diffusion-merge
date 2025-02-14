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




def merge_unets(unet1, unet2, weight1=0.5, weight2=0.5):
    """
    Merges two U-Net models by averaging their weights according to given contributions.
    """
    if weight1 + weight2 == 0:
        raise ValueError("Sum of weights must be nonzero.")


    merged_unet = unet1
    with torch.no_grad():
        for param1, param2 in zip(unet1.parameters(), unet2.parameters()):
            param1.data = weight1 * param1.data + weight2 * param2.data
    
    return merged_unet