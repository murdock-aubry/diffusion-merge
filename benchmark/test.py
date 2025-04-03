import torch
import numpy as np
from benchmark import *


data_link = "food_shard.parquet"
data_link = "/projects/dynamics/diffusion-tmp/data/test/" + data_link

num_prompts = -1

dataset = get_prompts_local(source = data_link, num_samples = num_prompts)
