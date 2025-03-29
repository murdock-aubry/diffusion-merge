import torch
from diffusers import UNet2DConditionModel
from utils import *
import gc

def setup_device():
    """Set up and return the device (GPU or CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clear_memory(aggressive=False):
    """Clear memory and CUDA cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if aggressive:
            # Force GPU synchronization and more aggressive memory cleanup
            torch.cuda.synchronize()
            
            if hasattr(torch.cuda, 'memory_stats'):
                torch.cuda.reset_peak_memory_stats()

def load_model_weights(model_name):
    """Load model weights from a pretrained model and extract different layer types."""
    print(f"Loading weights from {model_name}...", flush = True)
    torch.manual_seed(42)  # Set torch seed for reproducibility
    unet = load_unet(model_name).to("cpu")
    
    # Extract different types of weights and biases
    conv_weights = get_weights_key(unet, ["conv", "weight"], not_keys=["norm", "bias"])
    conv_biases = get_weights_key(unet, ["conv", "bias"], not_keys=["norm", "weight"])
    
    linear_weights = get_weights_key(unet, ["weight"], not_keys=["norm", "conv", "bias"])
    linear_biases = get_weights_key(unet, ["bias"], not_keys=["norm", "conv", "weight"])
    
    # Optionally extract norm weights (uncomment if needed)
    norm_weights = get_weights_key(unet, ["norm", "weight"], not_keys=["bias"])
    norm_biases = get_weights_key(unet, ["norm", "bias"], not_keys=["weight"])
    
    result = {
        'conv_weights'   : conv_weights,
        'conv_biases'    : conv_biases,
        'linear_weights' : linear_weights,
        'linear_biases'  : linear_biases,
        'norm_weights'   : norm_weights,
        'norm_biases'    : norm_biases
    }
    
    # Clean up
    del unet
    clear_memory()
    
    return result



def merge_conv_layers_multiple(weights, biases, nsamples, device, batch_size=10, thresh = 0.7):
    """Merge convolutional layers with batching to reduce memory usage."""
    new_weights = {}

    bias_names = biases[0].keys()
    
    # Get a list of keys that are in both weight dictionaries
    common_keys = set.intersection(*(set(weight.keys()) for weight in weights))
    
    for param_name in common_keys:
        print(f"Merging {param_name}. Expected shape: {weights[0][param_name].shape}.", flush = True)
        
        # Get the corresponding bias name
        bias_name = '.'.join(param_name.split('.')[:-1]) + ".bias"
        
        # Skip if the bias doesn't exist
        if bias_name not in biases[0]:
            continue
        
        # Process in batches to reduce memory usage

        params = []
        for weight in weights:
            params.append(weight[param_name].to(device))

        params_biases = []
        for bias in biases:
            params_biases.append(bias[bias_name].to(device))

        # Merge weights and biases
        new_param, new_bias = combine_conv_layers(params, params_biases, nsamples, device, thresh = thresh)
        
        # Store results on CPU to save GPU memory
        new_weights[param_name] = new_param.to("cpu")
        new_weights[bias_name] = new_bias.to("cpu")
        
        print(f"✓ Successfully merged {param_name}", flush = True)
        
        # Clean up GPU memory
        del params, params_biases, new_param, new_bias
        clear_memory()
    
    return new_weights


def merge_conv_layers(weights1, weights2, biases1, biases2, nsamples, device, batch_size=10, thresh = 0.7):
    """Merge convolutional layers with batching to reduce memory usage."""
    new_weights = {}
    
    # Get a list of keys that are in both weight dictionaries
    common_keys = set(weights1.keys()).intersection(set(weights2.keys()))
    
    for param_name in common_keys:
        print(f"Merging {param_name}. Expected shape: {weights1[param_name].shape}.", flush = True)
        
        # Get the corresponding bias name
        bias_name = '.'.join(param_name.split('.')[:-1]) + ".bias"
        
        # Skip if the bias doesn't exist
        if bias_name not in biases1 or bias_name not in biases2:
            continue
        
        # Process in batches to reduce memory usage
        param1 = weights1[param_name].to(device)
        param2 = weights2[param_name].to(device)
        bias1 = biases1[bias_name].to(device)
        bias2 = biases2[bias_name].to(device)
        
        # Merge weights and biases
        new_param, new_bias = combine_conv_layers([param1, param2], [bias1, bias2], nsamples, device, thresh = thresh)
        
        # Store results on CPU to save GPU memory
        new_weights[param_name] = new_param.to("cpu")
        new_weights[bias_name] = new_bias.to("cpu")
        
        print(f"✓ Successfully merged {param_name}", flush = True)
        
        # Clean up GPU memory
        del param1, param2, bias1, bias2, new_param, new_bias
        clear_memory()
    
    return new_weights

def merge_linear_layers_multiple(weights, biases, nsamples, device, batch_size=10, thresh = 0.7):
    # weights = [weight1 ... weight-n]
    # biases = [bias1 ... bias-n]

    new_weights = {}

    bias_names = biases[0].keys()

    common_keys = list(set(weights[0].keys()).intersection(*(set(weight.keys()) for weight in weights[1:])))
    total_keys = len(common_keys)

    for i in range(0, total_keys, batch_size):
        batch_keys = common_keys[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(total_keys+batch_size-1)//batch_size}, keys {i}-{min(i+batch_size-1, total_keys-1)}", flush = True)
        
        for param_name in batch_keys:
            # Get the corresponding bias name
            bias_name = '.'.join(param_name.split('.')[:-1]) + ".bias"
            
            # Skip if the bias doesn't exist
            # All models are of the same arch, no need to check all of them
            if bias_name not in bias_names:
                continue
            
            params = []
            for weight in weights: # load all weights for this batch
                params.append(weight[param_name].to(device, non_blocking=True))

            # Move to GPU with non-blocking transfer
            params_bias = []
            for i, bias in enumerate(biases):  # load all biases
                params_bias.append(bias[bias_name].to(device, non_blocking=True))
                # biases[i][bias_name] = biases[i][bias_name].to(device, non_blocking=True)

            # Handle convolutional linear weights (1x1 convs stored as linear)
            shape_flag = len(params[0].shape) > 2


            original_shape = params[0].shape if shape_flag else None
            
            if shape_flag:
                for ipar in range(len(params)):
                    params[ipar] = params[ipar][:, :, 0, 0]


            # Use non_blocking transfers and stream management for better GPU utilization
            with torch.cuda.stream(torch.cuda.Stream()):
                
                print(f"Merging {param_name}. Shape: {params[0].shape}.", flush = True)
                
                # Use chunking for extremely large parameters
                if params[0].numel() > 10_000_000:  # Approximately 400MB for float32
                    print(f"Large parameter detected ({params[0].numel()} elements). Processing in chunks...", flush = True)
                    chunks = 2  # Split into 4 chunks

                    new_param_chunks = []
                    new_bias_chunks = []
                    
                    # Process large matrices in chunks
                    rows_per_chunk = params[0].shape[0] // chunks
                    for chunk in range(chunks):
                        start_row = chunk * rows_per_chunk
                        end_row = start_row + rows_per_chunk if chunk < chunks-1 else params[0].shape[0]
                        
                        # Process each chunk
                        chunk_params = [param[start_row:end_row].clone() for param in params]
                        chunk_biases = [bias[start_row:end_row].clone() for bias in params_bias]
                        
                        # Only process bias for the last chunk
                        
                        chunk_new_param, chunk_new_bias = combine_linear_layers(
                            chunk_params, 
                            chunk_biases, 
                            nsamples // chunks, 
                            device,
                            thresh = thresh
                        )
                        
                        new_param_chunks.append(chunk_new_param.cpu())
                        new_bias_chunks.append(chunk_new_bias.cpu())
                        
                        # Clear memory after each chunk
                        del chunk_params, chunk_biases
                        clear_memory()
                    
                    # Combine chunks
                    new_param = torch.cat(new_param_chunks, dim=0)
                    new_bias = torch.cat(new_bias_chunks, dim=0)
                    del new_param_chunks, new_bias_chunks
                else:
                    # For normal sized parameters, process all at once
                    new_param, new_bias = combine_linear_layers(params, params_bias, nsamples, device, thresh = thresh)
                
                # Restore original shape if necessary
                if shape_flag:
                    new_param = new_param.view(*original_shape)
                
                # Store results
                new_weights[param_name] = new_param.cpu()
                new_weights[bias_name] = new_bias.cpu()
                
                print(f"✓ Successfully merged {param_name}", flush = True)
                
                # Clean up GPU memory
                del params, params_bias, new_param, new_bias
                clear_memory()
                
        # Make sure all GPU work is done before proceeding
        torch.cuda.synchronize()
        clear_memory()
        
    return new_weights

def merge_linear_layers(weights1, weights2, biases1, biases2, nsamples, device, batch_size=10, thresh = 0.7):
    """Merge linear layers with batching to reduce memory usage."""
    new_weights = {}
    
    # Get a list of keys that are in both weight dictionaries
    common_keys = list(set(weights1.keys()).intersection(set(weights2.keys())))
    total_keys = len(common_keys)
    
    # Process in batches of parameters
    for i in range(0, total_keys, batch_size):
        batch_keys = common_keys[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(total_keys+batch_size-1)//batch_size}, keys {i}-{min(i+batch_size-1, total_keys-1)}", flush = True)
        
        for param_name in batch_keys:
            # Get the corresponding bias name
            bias_name = '.'.join(param_name.split('.')[:-1]) + ".bias"
            
            # Skip if the bias doesn't exist
            if bias_name not in biases1 or bias_name not in biases2:
                continue
            
            param1 = weights1[param_name]
            param2 = weights2[param_name]
            
            # Handle convolutional linear weights (1x1 convs stored as linear)
            shape_flag = len(param1.shape) > 2
            original_shape = param1.shape if shape_flag else None
            
            if shape_flag:
                param1 = param1[:, :, 0, 0]
                param2 = param2[:, :, 0, 0]
            
            # Use non_blocking transfers and stream management for better GPU utilization
            with torch.cuda.stream(torch.cuda.Stream()):
                # Move to GPU with non-blocking transfer
                param1 = param1.to(device, non_blocking=True)
                param2 = param2.to(device, non_blocking=True)
                bias1 = biases1[bias_name].to(device, non_blocking=True)
                bias2 = biases2[bias_name].to(device, non_blocking=True)
                
                print(f"Merging {param_name}. Shape: {param1.shape}.", flush = True)
                
                # Use chunking for extremely large parameters
                if param1.numel() > 10_000_000:  # Approximately 400MB for float32
                    print(f"Large parameter detected ({param1.numel()} elements). Processing in chunks...", flush = True)
                    chunks = 2  # Split into 4 chunks
                    new_param_chunks = []
                    new_bias_chunks = []
                    
                    # Process large matrices in chunks
                    rows_per_chunk = param1.shape[0] // chunks
                    for chunk in range(chunks):
                        start_row = chunk * rows_per_chunk
                        end_row = start_row + rows_per_chunk if chunk < chunks-1 else param1.shape[0]
                        
                        # Process each chunk
                        chunk_param1 = param1[start_row:end_row].clone()
                        chunk_param2 = param2[start_row:end_row].clone()

                        chunk_bias1 = bias1[start_row:end_row].clone()
                        chunk_bias2 = bias2[start_row:end_row].clone()
                        
                        # Only process bias for the last chunk
                        
                        chunk_new_param, chunk_new_bias = combine_linear_layers(
                            [chunk_param1, chunk_param2], 
                            [chunk_bias1, chunk_bias2], 
                            nsamples // chunks, 
                            device,
                            thresh = thresh
                        )
                        
                        new_param_chunks.append(chunk_new_param.cpu())
                        new_bias_chunks.append(chunk_new_bias.cpu())
                        
                        # Clear memory after each chunk
                        del chunk_param1, chunk_param2, chunk_bias1, chunk_bias2
                        clear_memory()
                    
                    # Combine chunks
                    new_param = torch.cat(new_param_chunks, dim=0)
                    new_bias = torch.cat(new_bias_chunks, dim=0)
                    del new_param_chunks
                else:
                    # For normal sized parameters, process all at once
                    new_param, new_bias = combine_linear_layers([param1, param2], [bias1, bias2], nsamples, device, thresh = thresh)
                
                # Restore original shape if necessary
                if shape_flag:
                    new_param = new_param.view(*original_shape)
                
                # Store results
                new_weights[param_name] = new_param.cpu()
                new_weights[bias_name] = new_bias.cpu()
                
                print(f"✓ Successfully merged {param_name}", flush = True)
                
                # Clean up GPU memory
                del param1, param2, bias1, bias2, new_param, new_bias
                clear_memory()
                
        # Make sure all GPU work is done before proceeding
        torch.cuda.synchronize()
        clear_memory()
        
    return new_weights

def save_merged_weights(model_path, new_weights):
    """Save the merged weights to a model."""
    print(f"Loading base model to update with merged weights: {model_path}", flush = True)
    new_unet = UNet2DConditionModel.from_pretrained(
        "/w/383/murdock/models/unets/zipit/blank",
        torch_dtype=torch.float16 
    ).to("cpu")  # Start on CPU to save memory
    
    # Copy merged weights
    print(f"Updating model with {len(new_weights)} merged parameters...", flush = True)
    for name, param in new_weights.items():
        if name in new_unet.state_dict():
            new_unet.state_dict()[name].copy_(param)
        else:
            print(f"Warning: Parameter {name} not found in target model", flush = True)
    
    # Save model
    print(f"Saving model to {model_path}", flush = True)
    new_unet.save_pretrained(model_path)
    
    # Clean up
    del new_unet
    clear_memory()

def main():
    device = setup_device()
    print(f"Using device: {device}", flush = True)
    torch.manual_seed(62)  # Set torch random seed for reproducibility
    
    # Configuration
    nsamples = 100
    batch_size = 10
    thresh = 0.0

    unet1_name_read = "sd1.4"
    unet1_name = "CompVis/stable-diffusion-v1-4"

    unet2_name_read = "sd1.4-cocotuned"
    unet2_name = "frknayk/dreambooth_training"


    new_unet_name = f"/w/383/murdock/models/unets/zipit/{unet1_name_read}_{unet2_name_read}_thresh{thresh}"
    

    print("\n----- Phase 1: Merging Linear Layers -----", flush = True)
    # Reload models for linear layers

    with torch.no_grad():
        model1_weights = load_model_weights(unet1_name)
        model2_weights = load_model_weights(unet2_name)

        # Merge linear layers
        linear_merged_weights_multiple = merge_linear_layers_multiple(
            [model1_weights['linear_weights'], model2_weights['linear_weights']],
            [model1_weights['linear_biases'], model2_weights['linear_biases']],
            nsamples, 
            device,
            batch_size = batch_size,
            thresh = thresh
        )


        
        # Save linear results
        save_merged_weights(new_unet_name, linear_merged_weights)

        # Clean up final resources
        del model1_weights, model2_weights, linear_merged_weights
        clear_memory(aggressive=True)
    


    # Process models in phases to reduce memory footprint
    print("\n----- Phase 2: Merging Convolutional Layers -----", flush = True)
    
    # Load models
    with torch.no_grad():
        model1_weights = load_model_weights(unet1_name)
        model2_weights = load_model_weights(unet2_name)
        
        # Merge convolutional layers
        conv_merged_weights = merge_conv_layers_multiple(
            [model1_weights['conv_weights'], model2_weights['conv_weights']],
            [model1_weights['conv_biases'], model2_weights['conv_biases']],
            nsamples, 
            device,
            thresh = thresh
        )
        
        # Save intermediate results
        save_merged_weights(new_unet_name, conv_merged_weights)
    
        # Clear large dictionaries to free memory
        del model1_weights, model2_weights, conv_merged_weights
        clear_memory()

    print("\n----- Phase 3: Merging Normalization Layers -----", flush = True)




    with torch.no_grad():
        model1_weights = load_model_weights(unet1_name)
        model2_weights = load_model_weights(unet2_name)
        norm_merged_weights = combine_norm_layers(
            model1_weights['norm_weights'], 
            model2_weights['norm_weights'],
            model1_weights['norm_biases'], 
            model2_weights['norm_biases'],
            device
        )

    
        save_merged_weights(new_unet_name, norm_merged_weights)
        del model1_weights, model2_weights, norm_merged_weights
        clear_memory()

    print("\nModel merging completed successfully!", flush = True)

if __name__ == "__main__":
    main()