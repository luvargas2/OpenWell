import os
import torch
import numpy as np
from monai.data import ThreadDataLoader, load_decathlon_datalist, CacheDataset
from monai.data.meta_tensor import MetaTensor 

# --- SAFETY FOR PYTORCH 2.6+ ---
try:
    from numpy.core.multiarray import _reconstruct
    torch.serialization.add_safe_globals([_reconstruct, np.ndarray, MetaTensor])
except ImportError:
    try:
         torch.serialization.add_safe_globals([np.ndarray, MetaTensor])
    except:
         pass
# -------------------------------

def get_dataloaders(config, transforms, device):
    """
    Creates DataLoaders for Train, Val, and Test.
    """
    data_dir = config["data"]["data_dir"]
    split_json = config["data"]["split_json"]
    datasets = os.path.join(data_dir, split_json)
    
    train_transforms, val_transforms, test_transforms = transforms

    # 1. Load File Lists from JSON
    train_files = load_decathlon_datalist(datasets, True, "training")
    val_files = load_decathlon_datalist(datasets, True, "validation")
    
    print(f"[INFO] Dataset: {config['data']['dataset']}")
    print(f"[INFO] Train Samples: {len(train_files)}")
    print(f"[INFO] Val Samples: {len(val_files)}")

    # 2. Create Datasets (CacheDataset for Speed)
    # cache_rate=1.0 loads EVERYTHING into RAM. Reduce if you get OOM (e.g. 0.5)
    print("[INFO] Loading Training Data into RAM...")
    train_ds = CacheDataset(
        data=train_files, 
        transform=train_transforms, 
        cache_rate=1.0, 
        num_workers=8
    )
    
    print("[INFO] Loading Validation Data into RAM...")
    val_ds = CacheDataset(
        data=val_files, 
        transform=val_transforms, 
        cache_rate=1.0, 
        num_workers=4
    )
    
    # We reuse val_files for the 'test' loop (usually for one-shot enrollment checks)
    test_ds = CacheDataset(
        data=val_files, 
        transform=test_transforms, 
        cache_rate=1.0, 
        num_workers=0
    )

    # 3. Create Loaders
    train_loader = ThreadDataLoader(
        train_ds, 
        batch_size=config["training"]["batch_size"], 
        shuffle=True, 
        num_workers=0 # ThreadDataLoader manages workers internally
    )
    
    val_loader = ThreadDataLoader(
        val_ds, 
        batch_size=1, 
        shuffle=False, 
        num_workers=0
    )
    
    test_loader = ThreadDataLoader(
        test_ds, 
        batch_size=1, 
        shuffle=False, 
        num_workers=0
    )

    return train_loader, val_loader, test_loader