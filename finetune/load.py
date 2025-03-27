import pandas as pd
from datasets import load_dataset

categories = ["body-parts", "electronics", "clothes", "vehicles", "animals", "food", "text"]

path = f"output_shards/{categories[1]}_shard.parquet"

dataset = load_dataset('parquet', data_files=path, split = "train")

# Display the first few rows of the dataset
print(dataset[0].keys())



quit()