import torch
import json
from scipy import stats
import numpy as np
import os
import gc
import matplotlib.pyplot as plt 
from diffusers import DiffusionPipeline
from pltutils import *

def load_unet(path):
    pipeline = DiffusionPipeline.from_pretrained(
        path, 
        torch_dtype=torch.float16
    ).to("cuda")
    unet = pipeline.unet

    del pipeline

    return unet

def load_unet_local(path):
    pipeline = DiffusionPipeline.from_pretrained(
        path, 
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")
    unet = pipeline.unet

    del pipeline

    return unet


def get_layer_names(unet):
    return list(dict(unet.named_parameters()).keys())

def get_weights_key(unet, keys = [], not_keys = []):
    """
        Extracts only the weights within the model with keywords and excludes those with not_key

    """
    params = dict(unet.named_parameters())

    params_return = {}

    for name in params.keys():
        # must have all keys and no not keys
        if all(key in name for key in keys) and not any(not_key in name for not_key in not_keys):
            
            params_return[name] = params[name]
    return params_return




def get_weight(weight_name, unet):
    """
        Extracts the weight with a given name of a unet
    """

    # Get all named parameters from the unet
    params = dict(unet.named_parameters())
    
    # Extract the requested weight tensor
    if weight_name in params:
        return params[weight_name]
    else:
        raise ValueError(f"Weight '{weight_name}' not found in model parameters")
    


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


def compute_corr_slope_matrices(data, thresh):
    """
    Compute the correlation matrix and slope matrix for n datasets.
    
    Parameters:
    data (torch.Tensor): Tensor of shape [n, num_samples] containing n datasets 
                         each with num_samples data points.
    
    Returns:
    tuple: (correlation_matrix, slope_matrix) where both are torch.Tensor of shape [n, n]
    """
    # Compute means and center data
    means = data.mean(dim=1, keepdim=True)
    data -= means  # Modify in-place to save memory


    
    # scatter_plot(data[33], data[578], title = "", xlabel = "Filter 1", ylabel = "Filter 2", path_name = "outputs/conv_correlation1.png")
    

    # 

    # Compute variances and standard deviations
    variances = torch.mean(data**2, dim=1, keepdim=True)
    stds = torch.sqrt(variances)
    
    # Compute covariance matrix
    covariance_matrix = torch.matmul(data, data.t()) / data.shape[1]
    
    clear_memory(means, stds)
    
    # Compute correlation matrix
    std_outer = stds @ stds.t()
    correlation_matrix = covariance_matrix / std_outer


    ###### PLOTTING #######
    # data = data.to("cpu")
    # data = data.detach().numpy()
    # for i in range(30):
    #     for j in range(correlation_matrix.shape[0]):
    #         if torch.abs(correlation_matrix[i, j]) > 0.75 and torch.abs(correlation_matrix[i, j]) < 0.95 and correlation_matrix[i, j] < 0:
    #             print(f"({i}, {j}), {correlation_matrix[i, j]}")
    #             for k in range(j - 100, j + 100):
    #                 scatter_plot(data[i], data[k], title = "", xlabel = "", ylabel = "", path_name = f"gif/{k}.png")


    correlation_matrix[torch.abs(correlation_matrix) < thresh] = 0  # Set the entries which are below a reasonable minimal correlation threshold to 0    

    clear_memory(stds, std_outer)
    
    # Compute slope matrix
    slope_matrix = covariance_matrix / variances

    clear_memory(covariance_matrix, variances)
    
    return correlation_matrix, slope_matrix


def get_new_filters(filters_joint, coefs_mat, nmodels):

    num_filters = filters_joint.shape[0] // nmodels
    
    new_filters = torch.zeros_like(filters_joint[:num_filters])  # Avoid redundant shape lookup

    for ifilter in range(num_filters):
        new_filters[ifilter] = torch.sum(filters_joint * coefs_mat[ifilter, :, None, None, None], dim=0)
    
    return new_filters


def get_new_weight(weights_joint, coefs, nmodels):
    """
    Compute new weights based on correlation and slope matrices.
    
    Parameters:
    weights_joint (torch.Tensor): Joint weight matrix of shape [2n, m]
    corr_mat (torch.Tensor): Correlation matrix of shape [2n, 2n]
    slope_mat (torch.Tensor): Slope matrix of shape [2n, 2n]
    
    Returns:
    torch.Tensor: New weight matrix of shape [n, m]
    """
    nrows = weights_joint.shape[0] // nmodels
    
    new_weight = torch.zeros_like(weights_joint[:nrows])

    for irow in range(nrows):
        new_weight[irow] = torch.sum(weights_joint * coefs[irow, :, None], dim=0)
        # new_weight[irow] = (weights_joint[irow] + weights_joint[nrows + irow]) / 2

    return new_weight

def get_new_bias(biases, coef_mat, nmodels):

    nrows = biases.shape[0] // nmodels

    new_bias = coef_mat @ biases
    new_bias = new_bias[:nrows]

    return new_bias


def clear_memory(*values):
    for value in values:
        del value
    gc.collect()
    torch.cuda.empty_cache()
    return

def combine_linear_layers(weights, biases, nsamples, device, thresh = 0.7):
    """
    Combine multiple linear layers into a single equivalent layer.
    
    Parameters:
    weights (list of torch.Tensor): List of weight tensors.
    biases (list of torch.Tensor): List of bias tensors.
    nsamples (int): Number of random samples to generate.
    device (torch.device): Device to perform computations on.
    
    Returns:
    tuple: (new_weight, new_bias) with combined weights and biases.
    """

    nmodels = len(weights)

    input_dim = weights[0].shape[1]
    random_inputs = torch.randn(input_dim, nsamples, device=device, dtype=weights[0].dtype)
    
    outputs = []
    for weight in weights:
        output = weight @ random_inputs
        outputs.append(output)
        weight = weight.to("cpu")  # Move to CPU immediately after use
    clear_memory(random_inputs)
    
    outputs = torch.cat(outputs, dim=0)
    corr_mat, slope_mat = compute_corr_slope_matrices(outputs, thresh)
    clear_memory(outputs)  # Free memory
    
    corr_mat = abs(corr_mat) * slope_mat
    clear_memory(slope_mat)  # Free memory

    coefs = corr_mat / torch.norm(corr_mat, p=1, dim=0, keepdim=True)
    clear_memory(corr_mat)  # Free memory


    weights = torch.cat([weight.to(device) for weight in weights], dim = 0)
    weight = get_new_weight(weights, coefs, nmodels)
    clear_memory(weights)  # Free memory
    
    biases = torch.cat([bias.to(device) for bias in biases], dim = 0)
    # biases = torch.cat(biases, dim=0)
    bias = get_new_bias(biases, coefs, nmodels)
    clear_memory(biases, coefs)  # Free memory
    
    return weight, bias


def combine_norm_layers_multiple(weights, biases):

    new_weights = {}

    nmodels = len(weights)

    for name in weights[0].keys():
        new_weights[name] = sum(weight[name] for weight in weights) / nmodels

    for name in biases[0].keys():
        new_weights[name] = sum(bias[name] for bias in biases) / nmodels

    return new_weights


def combine_norm_layers(weights1, weights2, biases1, biases2, device):

    new_weights = {}

    for (name1, weight1), (name2, weight2) in zip(weights1.items(), weights2.items()):
        if name1 == name2:
            name = name1
        else:
            print("Name mismatch in normalization layers weights")

        new_weights[name] = (weight1 + weight2) / 2

    for (name1, weight1), (name2, weight2) in zip(biases1.items(), biases2.items()):
        if name1 == name2:
            name = name1
        else:
            print("Name mismatch in normalization layers biases")
            
        new_weights[name] = (weight1 + weight2) / 2

    return new_weights

def combine_weights(weights, nsamples, device, thresh = 0.7):
    input = weights[0].shape[1]

    random_inputs = torch.randn(input, nsamples, device=device, dtype=weights[0].dtype)

    outputs = [] 

    for weight in weights:
        output = weight @ random_inputs
        outputs.append(output)

    outputs = torch.cat(outputs, dim = 0)
    corr_mat, slope_mat = compute_corr_slope_matrices(outputs, thresh)
    
    weights_joint = torch.cat(weights, dim = 0)
    weight = get_new_weight(weights_joint, corr_mat, slope_mat)


    return weight

def combine_conv_layers(filters, biases, nsamples, device, thresh = 0.7):
    # filters = [filter1, filter2, ...] list of filters
    # collapses all filters to one which is the same size as the constituents

    nmodels = len(filters)

    in_channels = filters[0].shape[1]
    filter_size = filters[0].shape[-1]


    filters_joint = torch.cat(filters, dim = 0)

    random_filters = torch.randn(nsamples, in_channels, filter_size, filter_size, device=device, dtype=filters[0].dtype)

    filter_vals = []

    for filter in filters:
        filter_out = get_filter_samples(filter, random_filters, nsamples)
        filter_vals.append(filter_out)

    filter_vals = torch.cat(filter_vals, dim = 0)

    corr_mat, slope_mat = compute_corr_slope_matrices(filter_vals, thresh)
    coefs = abs(corr_mat) * slope_mat
    coefs = coefs / torch.norm(coefs, p=1, dim=0, keepdim=True)

    new_filters = get_new_filters(filters_joint, coefs, nmodels)

    biases_joint = torch.cat(biases, dim = 0)
    new_bias = get_new_bias(biases_joint, coefs, nmodels)

    return new_filters, new_bias

