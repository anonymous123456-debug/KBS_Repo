from datasets import load_dataset

ds = load_dataset("./winograd")
print(ds['test'][0])