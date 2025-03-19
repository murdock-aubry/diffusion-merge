import torch
import json
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt 
from diffusers import DiffusionPipeline

def load_json_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
    

def load_unet(path):
    pipeline = DiffusionPipeline.from_pretrained(
        path, 
        torch_dtype=torch.float16
    ).to("cuda")
    unet = pipeline.unet

    del pipeline

    return unet

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


def get_layer_names(unet):

    layers = []
    
    for name, param in unet.named_parameters():
        layers.append(name)


        # if "conv" not in name:
        #     print(name, param.shape)

        # print(param.shape, name)

        # if "conv" in name:
        #     print(name, param.shape)

    return layers




def get_weight(weight_name, unet):
    # Get all named parameters from the unet
    params = dict(unet.named_parameters())
    
    # Extract the requested weight tensor
    if weight_name in params:
        return params[weight_name]
    else:
        raise ValueError(f"Weight '{weight_name}' not found in model parameters")



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


def get_filter_samples(filters, random_samples, nsamples):

    out_channels = filters.shape[0]

    filter_vals = []

    for ifilter in range(out_channels):
    
        filter = filters[ifilter]

        outs = []

        for isample in range(nsamples):

            val = torch.sum(filter * random_samples[isample])
            
            outs.append(val)

        filter_vals.append(torch.stack(outs, dim = 0))

    filter_vals = torch.stack(filter_vals, dim = 0)

    return filter_vals


def compute_corr_slope_matrices(data):
    """
    Compute the correlation matrix and slope matrix for n datasets.
    
    Parameters:
    data (torch.Tensor): Tensor of shape [n, num_samples] containing n datasets 
                         each with num_samples data points.
    
    Returns:
    tuple: (correlation_matrix, slope_matrix) where both are torch.Tensor of shape [n, n]
    """
    # Get dimensions
    n = data.shape[0]
    num_samples = data.shape[1]
    
    # Compute means
    means = data.mean(dim=1, keepdim=True)
    
    # Center the data (subtract mean)
    centered_data = data - means
    
    # Compute correlation matrix
    # For correlation, we need normalized data
    variances = torch.mean(centered_data**2, dim=1)
    stds = torch.sqrt(variances)
    
    # Compute all pairs of covariances efficiently
    # cov(X,Y) = E[(X-E[X])(Y-E[Y])]
    covariance_matrix = torch.matmul(centered_data, centered_data.t()) / num_samples
    
    # Compute correlation matrix
    # corr(X,Y) = cov(X,Y) / (std(X) * std(Y))
    # Using outer product of stds to get denominator matrix
    std_outer = torch.outer(stds, stds)
    correlation_matrix = covariance_matrix / std_outer
    
    # Compute slope matrix
    # slope(X→Y) = cov(X,Y) / var(X)
    # We can use broadcasting to divide each row of covariance matrix by variances
    slope_matrix = covariance_matrix / variances.unsqueeze(1)
    
    return correlation_matrix, slope_matrix



def get_new_filters(filters_joint, corr_mat, slope_mat):

    num_filters = filters_joint.shape[0] // 2

    original_filters = filters_joint[:num_filters]

    coefs = abs(corr_mat) * slope_mat
    normalized_coefs = coefs / torch.norm(coefs, dim = 0)

    
    new_filters = torch.zeros((original_filters.shape))

    for ifilter in range(num_filters):

        weighted_filters = filters_joint * normalized_coefs[ifilter].view(-1, 1, 1, 1)
        new_filter = torch.sum(weighted_filters, dim = 0)
        new_filters[ifilter] = new_filter

    return new_filters


def get_new_weight(weights_joint, corr_mat, slope_mat):

    nrows = weights_joint.shape[0] // 2

    original_weight = weights_joint[:nrows]

    coefs = abs(corr_mat) * slope_mat
    normalized_coefs = coefs / torch.norm(coefs, dim = 0)

    new_weight = torch.zeros((original_weight.shape))

    for irow in range(nrows):

        weighted_rows = weights_joint * normalized_coefs[irow].view(-1, 1)
        new_row = torch.sum(weighted_rows, dim = 0)
        new_weight[irow] = new_row

    return new_weight


def get_new_bias(biases, coef_mat):



    nrows = biases.shape[0] // 2

    new_bias = coef_mat @ biases
    new_bias = new_bias[:nrows]

    print(new_bias.shape)

    return new_bias

def combine_linear_layers(weights, biases, nsamples, device):

    input = weights[0].shape[1]

    random_inputs = torch.randn(input, nsamples, device=device, dtype=weights[0].dtype)

    outputs = [] 

    for weight in weights:
        output = weight @ random_inputs
        outputs.append(output)

    outputs = torch.cat(outputs, dim = 0)
    corr_mat, slope_mat = compute_corr_slope_matrices(outputs)
    coefs = abs(corr_mat) * slope_mat
    normalized_coefs = coefs / torch.norm(coefs, dim = 0)
    
    
    weights_joint = torch.cat(weights, dim = 0)
    weight = get_new_weight(weights_joint, corr_mat, slope_mat)

    biases_joint = torch.cat(biases, dim = 0)
    bias = get_new_bias(biases_joint, normalized_coefs)

    return weight, bias

def combine_weights(weights, nsamples, device):
    input = weights[0].shape[1]

    random_inputs = torch.randn(input, nsamples, device=device, dtype=weights[0].dtype)

    outputs = [] 

    for weight in weights:
        output = weight @ random_inputs
        outputs.append(output)

    outputs = torch.cat(outputs, dim = 0)
    corr_mat, slope_mat = compute_corr_slope_matrices(outputs)
    coefs = abs(corr_mat) * slope_mat
    normalized_coefs = coefs / torch.norm(coefs, dim = 0)
    
    
    weights_joint = torch.cat(weights, dim = 0)
    weight = get_new_weight(weights_joint, corr_mat, slope_mat)

    return weight

def combine_conv_layers(filters, nsamples, device):
    # filters = [filter1, filter2, ...] list of filters
    # collapses all filters to one which is the same size as the constituents

    in_channels = filters[0].shape[1]
    filter_size = filters[0].shape[-1]


    filters_joint = torch.cat(filters, dim = 0)

    random_filters = torch.randn(nsamples, in_channels, filter_size, filter_size, device=device, dtype=filters[0].dtype)

    filter_vals = []

    for filter in filters:
        filter_out = get_filter_samples(filter, random_filters, nsamples)
        filter_vals.append(filter_out)

    filter_vals = torch.cat(filter_vals, dim = 0)

    corr_mat, slope_mat = compute_corr_slope_matrices(filter_vals)
    new_filters = get_new_filters(filters_joint, corr_mat, slope_mat)

    return new_filters