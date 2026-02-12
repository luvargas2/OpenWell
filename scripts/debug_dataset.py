import sys
import os
import torch
import yaml

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loader import get_dataloaders
from src.data.transforms import get_transforms

def verify_data():
    # Load your config (adjust path if needed)
    config_path = "configs/config_btcv.yaml" 
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cpu") # CPU is fine for checking labels
    
    # Get Transforms & Loaders
    transforms = get_transforms(config["data"]["dataset"], device)
    train_loader, _, _ = get_dataloaders(config, transforms, device)
    
    print("\n[INFO] Checking first batch for '255' labels...")
    batch = next(iter(train_loader))
    labels = batch["label"]
    
    unique_vals = torch.unique(labels)
    print(f"Unique Labels found: {unique_vals.tolist()}")
    
    if 255 in unique_vals:
        print("SUCCESS: Unseen classes are mapped to 255.")
    else:
        print("WARNING: No 255 found in this batch. Try running again or check.")

if __name__ == "__main__":
    verify_data()