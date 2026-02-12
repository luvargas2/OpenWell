import sys
import os
import yaml
import torch
import argparse
import numpy as np
import warnings

# Add project root to python path so 'src' is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from monai.utils import set_determinism
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data.loader import get_dataloaders
from src.data.transforms import get_transforms
from src.models.swin_unetr import get_medopenseg
from src.models.memory_bank import MemoryBankV
from src.training.losses import OpenSetDiceCELoss
from src.training.trainer import Trainer

# --- SAFETY FOR PYTORCH 2.6+ ---
try:
    from monai.data.meta_tensor import MetaTensor
    from numpy.core.multiarray import _reconstruct
    torch.serialization.add_safe_globals([_reconstruct, np.ndarray, MetaTensor])
except:
    pass
# -------------------------------

warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)
    
    set_determinism(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # 1. Data
    transforms = get_transforms(config["data"]["dataset"], device)
    train_loader, val_loader, _ = get_dataloaders(config, transforms, device)

    # 2. Model
    embed_dim = config["model"].get("embed_dim_final", 128) 
    model = get_medopenseg(
        device=device,
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        img_size=(96,96,96),
        feature_size=config["model"]["feature_size"],
        embed_dim_final=embed_dim, 
        pre_trained_weights=config["training"].get("pretrained_weights"),
    )
    
    # 3. Memory Bank
    memory_bank = None
    if config["training"].get("use_memory_bank", False):
        print("[INFO] Initializing Memory Bank...")
        memory_bank = MemoryBankV(
            feature_dim=embed_dim,
            memory_size=config["training"]["memory_size"],
            save_path=os.path.join(config["training"]["checkpoint_dir"], "prototypes")
        ).to(device)

    # 4. Optimization
    # ignore_index=255 matches the transform logic
    loss_fn = OpenSetDiceCELoss(ignore_index=255)
    
    lr = float(config["training"]["lr"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = CosineAnnealingLR(optimizer, T_max=config["training"]["max_iterations"], eta_min=1e-6)

    # 5. Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        memory_bank=memory_bank,
        device=device,
        config=config,
        scheduler=scheduler
    )

    trainer.fit()

if __name__ == "__main__":
    main()