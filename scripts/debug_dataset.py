import sys
import os
import torch
import yaml
import numpy as np

# Add project root to python path so 'src' is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loader import get_dataloaders
from src.data.transforms import get_transforms

def verify_amos_data():
    # 1. Load AMOS Config
    config_path = "configs/config_amos.yaml"
    
    if not os.path.exists(config_path):
        print(f"[ERROR] Config file not found: {config_path}")
        return

    print(f"[INFO] Loading config: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Force dataset name to confirm we are using AMOS transforms
    dataset_name = config["data"]["dataset"]
    print(f"[INFO] Dataset Name: {dataset_name}")
    
    device = torch.device("cpu") # CPU is fine for checking labels
    
    # 2. Get Transforms & Loaders
    print("[INFO] Initializing Transforms...")
    # Note: ensure get_transforms in src/data/transforms.py handles "AMOS"
    transforms = get_transforms(dataset_name, device)
    
    print("[INFO] Initializing DataLoader...")
    train_loader, _, _ = get_dataloaders(config, transforms, device)
    
    # 3. Check the first few batches
    # We loop because 255 might not appear in *every* crop if the organ is small
    print("\n[INFO] Checking batches for '255' labels...")
    found_255 = False
    
    for i, batch in enumerate(train_loader):
        if i >= 10: break # Check up to 10 batches
        
        labels = batch["label"]
        unique_vals = torch.unique(labels)
        print(f"Batch {i}: Unique Labels -> {unique_vals.tolist()}")
        
        if 255 in unique_vals:
            found_255 = True
            print(f"  >>> FOUND IT! Batch {i} contains the hidden class (255).")
            break
    
    print("-" * 30)
    if found_255:
        print(" SUCCESS: Unseen classes are correctly mapped to 255.")
    else:
        print(" WARNING: No 255 found in first 10 batches.")
        print("   Possibilities:")
        print("   1. The 'unseen_ids' list in transforms.py is empty or wrong.")
        print("   2. The crop size is too small and missed the organ.")
        print("   3. The dataset variable in config_amos.yaml is not 'AMOS'.")

if __name__ == "__main__":
    verify_amos_data()