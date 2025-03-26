import datasets as ds


data_name = "reach-vb/pokemon-blip-captions"

dataset = ds.load_dataset(
    data_name
)


print(dataset["train"]["image"])