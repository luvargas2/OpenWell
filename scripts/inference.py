import torch
import os
import argparse
import numpy as np
import yaml
import sys
import scipy.ndimage as ndimage

import matplotlib.pyplot as plt
import nibabel as nib
from monai.inferers import sliding_window_inference
from monai.data import load_decathlon_datalist, Dataset, ThreadDataLoader
from monai.data.meta_tensor import MetaTensor

# --- FIX FOR PYTORCH 2.6+ WEIGHTS_ONLY ERROR ---
torch.serialization.add_safe_globals([MetaTensor])

# Add project root to python path so 'src' is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# --- YOUR IMPORTS ---
from src.models.memory_bank import MemoryBankV
from src.models.swin_unetr import get_medopenseg
from src.data.transforms import get_btcv_transforms, get_brats_transforms, get_amos_transforms, get_msdpancreas_transforms

def load_transforms(config, device):
    dataset = config["data"]["dataset"]
    if dataset == "BRATS": return get_brats_transforms(device)
    elif dataset == "BTCV": return get_btcv_transforms(device)
    elif dataset == "AMOS": return get_amos_transforms(device)
    elif dataset == "MSD_PANCREAS": return get_msdpancreas_transforms(device)
    else: raise ValueError(f"Unknown dataset: {dataset}")

def find_slice_with_most_unseen(label, unseen_classes=[11, 12, 13]):
    mask = np.isin(label, unseen_classes)
    unseen_pixels = np.sum(mask, axis=(0, 1))
    if unseen_pixels.max() == 0:
        return label.shape[2] // 2 
    return np.argmax(unseen_pixels)

# ==========================================
# INTERACTIVE ENROLLMENT FUNCTIONS
# ==========================================
def simulate_user_click(label_np, unseen_classes=[11, 12, 13]):
    """Simulates a user clicking on the center of a novel anomaly."""
    mask = np.isin(label_np, unseen_classes)
    if not np.any(mask):
        return None
    coords = ndimage.center_of_mass(mask)
    return tuple(int(c) for c in coords)

def create_guidance_map(shape, click_coord, sigma=6.0):
    """Generates a 3D Gaussian heatmap centered at the user click (Eq. 9)."""
    z, y, x = np.indices(shape)
    cz, cy, cx = click_coord
    dist_sq = (z - cz)**2 + (y - cy)**2 + (x - cx)**2
    heatmap = np.exp(-dist_sq / (2 * sigma**2))
    return torch.tensor(heatmap, dtype=torch.float32)

def enroll_new_class(embedding, guidance_map, device, default_kappa=50.0):
    """Calculates mu_new and kappa_new based on the guided local features."""
    g = guidance_map.to(device).unsqueeze(0).unsqueeze(0)
    
    # Aggregate features weighted by the guidance map
    weighted_features = embedding * g
    sum_features = weighted_features.sum(dim=(2, 3, 4)) # [1, C]
    sum_g = g.sum() + 1e-8
    
    # Calculate Mean Direction (mu_c)
    mu_new = sum_features / sum_g
    mu_new = mu_new / (torch.norm(mu_new, p=2, dim=1, keepdim=True) + 1e-8)
    
    # Assign Concentration (kappa_c)
    kappa_new = torch.tensor([default_kappa], device=device)
    
    return mu_new, kappa_new

# ==========================================
# VISUALIZATION
# ==========================================
def visualize_interactive_enrollment(inputs, labels, pred_seg, open_pred, guidance_map, click_coord, slice_idx, output_path):
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    vol_slice = np.rot90(inputs[:, :, slice_idx])
    lbl_slice = np.rot90(labels[:, :, slice_idx])
    pred_slice = np.rot90(pred_seg[:, :, slice_idx])
    open_slice = np.rot90(open_pred[:, :, slice_idx])
    unseen_gt_mask = np.isin(lbl_slice, [11, 12, 13])
    
    # 1. Ground Truth + Simulated Click
    axes[0].imshow(vol_slice, cmap="gray")
    masked_lbl = np.ma.masked_where(lbl_slice == 0, lbl_slice)
    axes[0].imshow(masked_lbl, cmap="turbo", alpha=0.6)
    axes[0].contour(unseen_gt_mask, colors='lime', linewidths=1.2, linestyles='dashed')
    
    if click_coord and click_coord[2] == slice_idx:
        axes[0].plot(click_coord[0], click_coord[1], marker='+', color='white', markersize=15, markeredgewidth=2)
    
    axes[0].set_title("Ground Truth\n(+ Simulated Click)")
    axes[0].axis("off")
    
    # 2. Closed-Set Prediction (Memory Bank Query 1)
    axes[1].imshow(pred_slice, cmap="turbo")
    axes[1].set_title("Closed-Set Prediction\n(Fails on Unseen)")
    axes[1].axis("off")
    
    # 3. Guidance Map
    if guidance_map is not None:
        guidance_slice = np.rot90(guidance_map[:, :, slice_idx].cpu().numpy())
        axes[2].imshow(vol_slice, cmap="gray")
        axes[2].imshow(guidance_slice, cmap="cool", alpha=0.6)
    axes[2].set_title("User Guidance Map $\mathcal{G}$\n(Local Feature Extraction)")
    axes[2].axis("off")

    # 4. Open-World Enrolled Result (Memory Bank Query 2)
    axes[3].imshow(open_slice, cmap="turbo")
    axes[3].contour(open_slice == 99, colors='red', linewidths=1.5) 
    axes[3].set_title("Instant Open-World Result\n(New Well Dug: Red Outline)")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# ==========================================
# MAIN INFERENCE LOOP
# ==========================================
def load_model(checkpoint_path, device, config):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = get_medopenseg(
        device=device, in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"], img_size=(96,96,96),
        feature_size=config["model"]["feature_size"], embed_dim_final=config["model"]["embed_dim_final"],
        pre_trained_weights=None
    )
    state_dict = checkpoint["model_state_dict"]
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model

def infer_and_segment(model, memory_bank, test_loader, device, output_dir):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    results = [] 
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            filename = f"sample_{i}"
            if "image_meta_dict" in batch:
                filename = os.path.basename(batch["image_meta_dict"]["filename_or_obj"][0]).split('.')[0]

            print(f"\n[PROC] Processing {filename}...")
            
            # --- PHASE 1: Feature Extraction ---
            with torch.amp.autocast(device.type):
                _, embedding = sliding_window_inference(
                    inputs, roi_size=(96, 96, 96), sw_batch_size=4, overlap=0.5,
                    mode="gaussian", predictor=model
                )
            
            # CRITICAL THEORETICAL FIX: 
            # We discard the neural network's linear head entirely. 
            # The Memory Bank IS the open-set classifier!
            _, pred_seg = memory_bank.query_voxelwise_novelty(embedding, include_background=True)
            
            # --- PHASE 2: Instant Interactive Enrollment ---
            vol_np = inputs[0, 0].cpu().numpy()
            lbl_np = labels[0, 0].cpu().numpy()
            pred_np = pred_seg[0].cpu().numpy()
            
            open_world_pred = pred_seg.clone()
            guidance_map = None
            
            # Check if class 99 is already in the memory bank from a previous patient!
            already_enrolled = (99 in memory_bank.prototypes)

            click_coord = simulate_user_click(lbl_np)
            
            if click_coord and not already_enrolled:
                print(f"[INFO] Novel anomaly detected! Simulating user click at {click_coord}")
                
                # 1. Generate Gaussian guidance
                guidance_map = create_guidance_map(lbl_np.shape, click_coord, sigma=6.0).to(device)
                
                # 2. Extract local features
                mu_new, kappa_new = enroll_new_class(embedding, guidance_map, device)
                
                # 3. Dig the new well permanently in the Memory Bank!
                memory_bank.enroll_interactive_prototype(
                    new_class_id=99, 
                    mu_new=mu_new.squeeze(0), # Remove batch dim [F]
                    kappa_new=kappa_new.squeeze(0) # [1]
                )
                
                # 4. MAGIC STEP: Just query the memory bank again!
                # Because the new well exists, the math automatically updates the segmentation.
                _, open_world_pred = memory_bank.query_voxelwise_novelty(embedding, include_background=True)
                
            elif already_enrolled:
                print("[INFO] Class 99 was enrolled on a previous patient. It will segment automatically!")
            else:
                print("[INFO] No anomaly found in this volume.")

            # --- VISUALIZATION ---
            slice_idx = find_slice_with_most_unseen(lbl_np)
            vis_path = os.path.join(output_dir, f"{filename}_interactive.png")
            
            visualize_interactive_enrollment(
                inputs=vol_np, labels=lbl_np, pred_seg=pred_np,
                open_pred=open_world_pred[0].cpu().numpy(),
                guidance_map=guidance_map, click_coord=click_coord,
                slice_idx=slice_idx, output_path=vis_path
            )
            
            results.append(open_world_pred[0].cpu().numpy())

    return results 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_btcv")
    parser.add_argument("--exp", type=str, default="btcv/memory_enc3")
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config_path = os.path.join("./configs", f"{args.config}.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        
    exps_root = '/home/vargas/openwell/outputs'
    checkpoint_path = os.path.join(exps_root, args.exp, 'best_checkpoint.pth')

    model = load_model(checkpoint_path, device, config)
    
    embed_dim = config["training"].get("embed_dim_final", 128)
    memory_bank = MemoryBankV(memory_size=100, feature_dim=embed_dim).to(device)
    memory_bank.load_memory_bank(os.path.join(exps_root, args.exp, "energy_memory_bank.pth"), device=device)

    data_dir = config["data"]["data_dir"]
    datasets = os.path.join(data_dir, config["data"]["split_json"])
    _, _, test_transforms = load_transforms(config, device)
    
    test_files = load_decathlon_datalist(datasets, True, "validation")[:5] 
    test_ds = Dataset(data=test_files, transform=test_transforms)
    test_loader = ThreadDataLoader(test_ds, batch_size=1, num_workers=0)

    output_vis_dir = os.path.join(exps_root, args.exp, "vis_interactive")
    results = infer_and_segment(model, memory_bank, test_loader, device, output_vis_dir)
    
    if not results: return

    output_seg_dir = "output_segmentations"
    os.makedirs(output_seg_dir, exist_ok=True)
    for i, seg in enumerate(results):
        nib.save(nib.Nifti1Image(seg.astype(np.int16), affine=np.eye(4)), os.path.join(output_seg_dir, f"segmentation_{i}.nii.gz"))

if __name__ == "__main__":
    main()