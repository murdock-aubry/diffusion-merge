import torch
from utils import *

# param_name = "mid_block.resnets.1.conv1.weight"
param_name = "up_blocks.1.resnets.0.conv1.weight"
param_name = "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k.weight"

param_name = "up_blocks.2.attentions.2.proj_out.weight" 
bias_name = "up_blocks.2.attentions.2.proj_out.bias"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


unet = "CompVis/stable-diffusion-v1-4"
unet = load_unet(unet)



# Convolutions have shape [out_channel, in_channel, 3, 3], or [num_filters, num_channels, (filter shape)]

layer_names = get_layer_names(unet)

nlayers = len(layer_names)

print("Merging convolutional layers")
for name in layer_names:
    if "conv" in name and "weight" in name:
        weight_name = name
        layer_name = '.'.join(name.split('.')[:-1])
        bias_name = f"{layer_name}.bias"
        if bias_name in layer_names:
            layer_names.remove(weight_name)
            layer_names.remove(bias_name)

print("Merging linear layers with baises")
for name in layer_names[:]:
    if "weight" in name and ".norm" not in name:
        # weight_name = name


        layer_name = '.'.join(name.split('.')[:-1])
        bias_name = f"{layer_name}.bias"

        if bias_name in layer_names:
            layer_names.remove(name)
            layer_names.remove(bias_name)


print("Merging linear layers WITHOUT biases")
for name in layer_names[:]:  # Create a shallow copy of layer_names
    if "weight" in name and ".norm" not in name:
        layer_names.remove(name)



print("Linearly interpolating the layer norm weights and biases")
for name in layer_names[:]:
    if "norm" in name and "weight" in name:

        layer_name = '.'.join(name.split('.')[:-1])
        bias_name = f"{layer_name}.bias"

        if bias_name in layer_names:
            layer_names.remove(name)
            layer_names.remove(bias_name)



print(len(layer_names))

quit()
    






param1 = get_weight(param_name, unet)
bias1 = get_weight(bias_name, unet)
param1 = param1[:, :, 0, 0]


quit()
del unet
torch.cuda.empty_cache()


# unet = "OFA-Sys/small-stable-diffusion-v0"
# unet = "ImageInception/ArtifyAI-v1.0"



unet = load_unet(unet)
param2 = get_weight(param_name, unet)
param2 = param2[:, :, 0, 0]
bias2 = get_weight(bias_name, unet)

del unet
torch.cuda.empty_cache()


output = param1.shape[0]
input = param1.shape[1]

nsamples = 100


weight, bias = combine_linear_layers([param1, param2], [bias1, bias2], nsamples, device)



# print(input, output)






# outputs = torch.cat((outputs1, outputs2), dim = 0)


# corr_mat, slope_mat = compute_corr_slope_matrices(outputs)


# joint_param = torch.cat((param1, param2), dim = 0)


# new_weight = get_new_weight(joint_param, corr_mat, slope_mat)

# print(new_weight.shape)
# # print(random_filters.shape)



quit()









nsamples = 100
new_filters = combine_conv_layers([param1, param2], nsamples, device)