import torch
from diffusers import UNet2DConditionModel
from utils import *
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

nsamples = 100

unet1_name = "CompVis/stable-diffusion-v1-4"
unet2_name = "frknayk/dreambooth_training"


new_unet_name = "/w/383/murdock/models/unets/zipit/model1"


# Load conv weights
unet1 = load_unet(unet1_name).to("cpu")
conv_weights1 = get_weights_key(unet1, ["conv", "weight"], not_keys=["norm", "bias"])
conv_biases1 = get_weights_key(unet1, ["conv", "bias"], not_keys=["norm", "weight"])

del unet1
gc.collect()
torch.cuda.empty_cache()


unet2 = load_unet(unet2_name).to("cpu")
conv_weights2 = get_weights_key(unet2, ["conv", "weight"], not_keys=["norm", "bias"])
conv_biases2 = get_weights_key(unet2, ["conv", "bias"], not_keys=["norm", "weight"])

del unet2
gc.collect()
torch.cuda.empty_cache()



new_weights = {}

idx = 0

# merge conv layers
for (name1, param1), (name2, param2) in zip(conv_weights1.items(), conv_weights2.items()):

    param1 = param1.to("cuda") 
    param2 = param2.to("cuda") 
    
    idx += 1

    if idx > 1:
        continue
    
    print(f"merging {name1}. Expected shaped: {param1.shape}.")

    param_name = name1

    bias_name = '.'.join(name1.split('.')[:-1]) + ".bias"

    if bias_name in list(conv_biases1.keys()):

        bias1 = conv_biases1[bias_name]
        bias2 = conv_biases2[bias_name]

        bias1 = bias1.to("cuda") 
        bias2 = bias2.to("cuda")
        
        new_param, new_bias = combine_conv_layers([param1, param2], [bias1, bias2], nsamples, device)

        new_param = new_param.to("cpu")
        new_bias = new_bias.to("cpu")

        new_weights[param_name] = new_param
        new_weights[bias_name] = new_bias

        print("Shapes match:", param1.shape == new_param.shape)

        # Delete parameters to free up CUDA memory
        del param1, param2, bias1, bias2, new_param, new_bias
        gc.collect()
        torch.cuda.empty_cache()

            
del conv_weights1, conv_weights2, conv_biases1, conv_biases2
gc.collect()
torch.cuda.empty_cache()


new_unet = UNet2DConditionModel.from_pretrained(
    new_unet_name,
    torch_dtype=torch.float16 
).to("cuda")

layer_names = get_layer_names(new_unet)


for (name, param) in new_weights.items():
    new_unet.state_dict()[name].copy_(param)

new_unet.save_pretrained(new_unet_name)

del new_unet, new_weights
gc.collect()
torch.cuda.empty_cache()







# Load models on CPU to save GPU memory
unet1 = load_unet(unet1_name).to("cpu")
linear_weights1 = get_weights_key(unet1, ["weight"], not_keys=["norm", "conv", "bias"])
linear_biases1 = get_weights_key(unet1, ["bias"], not_keys=["norm", "conv", "weight"])
del unet1
gc.collect()

unet2 = load_unet(unet2_name).to("cpu")
linear_weights2 = get_weights_key(unet2, ["weight"], not_keys=["norm", "conv", "bias"])
linear_biases2 = get_weights_key(unet2, ["bias"], not_keys=["norm", "conv", "weight"])
del unet2
gc.collect()

new_weights = {}

# Merge layers efficiently
for name1, param1 in linear_weights1.items():
    param2 = linear_weights2[name1]

    # Flatten conv layers for processing
    shape_flag = len(param1.shape) > 2
    if shape_flag:
        param1 = param1[:, :, 0, 0]
        param2 = param2[:, :, 0, 0]

    # Move to GPU only when necessary
    param1, param2 = param1.to("cuda", non_blocking=True), param2.to("cuda", non_blocking=True)

    print(f"Merging {name1}. Expected shape: {param1.shape}.")

    bias_name = ".".join(name1.split(".")[:-1]) + ".bias"
    if bias_name in linear_biases1:
        bias1, bias2 = linear_biases1[bias_name].to("cuda", non_blocking=True), linear_biases2[bias_name].to("cuda", non_blocking=True)

        # Merge weights and biases
        new_param, new_bias = combine_linear_layers([param1, param2], [bias1, bias2], nsamples, "cuda")

        # Restore conv layer shape if necessary
        if shape_flag:
            new_param = new_param[:, :, None, None]

        # Move results back to CPU
        new_weights[name1] = new_param.to("cpu", non_blocking=True)
        new_weights[bias_name] = new_bias.to("cpu", non_blocking=True)

        # Free GPU memory
        del param1, param2, bias1, bias2, new_param, new_bias
        torch.cuda.empty_cache()

# Free CPU memory
del linear_weights1, linear_weights2, linear_biases1, linear_biases2
gc.collect()

# Load new UNet model
new_unet = UNet2DConditionModel.from_pretrained(new_unet_name, torch_dtype=torch.float16).to("cuda")
layer_names = get_layer_names(new_unet)

# Copy merged weights
for name, param in new_weights.items():
    new_unet.state_dict()[name].copy_(param)

# Save model
new_unet.save_pretrained(new_unet_name)

# Cleanup
del new_unet, new_weights
torch.cuda.empty_cache()

quit()




# Convolutions have shape [out_channel, in_channel, 3, 3], or [num_filters, num_channels, (filter shape)]

# merge linear layers
conv_weights = get_weights_key(unet1, ["conv", "weight"], not_keys=["norm", "bias"])
conv_biases = get_weights_key(unet1, ["conv", "bias"], not_keys=["norm", "weight"])

linear_weights = get_weights_key(unet1, ["weight"], not_keys=["norm", "conv", "bias"])
linear_biases = get_weights_key(unet1, ["bias"], not_keys=["norm", "conv", "weight"])

norm_weights = get_weights_key(unet1, ["norm", "weight"], not_keys=["bias"])
norm_biases = get_weights_key(unet1, ["norm", "bias"], not_keys=["weight"])

