import torch
from src.data.loader import get_loader
from src.configs import load_config

def verify_data():
    config = load_config("configs/config_btcv.yaml")
    loader = get_loader(config, split="train")
    
    batch = next(iter(loader))
    labels = batch["label"]
    
    unique_vals = torch.unique(labels)
    print(f"Unique Labels in Batch: {unique_vals}")
    
    if 255 in unique_vals:
        print("SUCCESS: 'Unseen' classes are correctly mapped to 255 (Ignore Index).")
    else:
        print("FAILURE: No 255 found. Transforms are not applied correctly!")

if __name__ == "__main__":
    verify_data()