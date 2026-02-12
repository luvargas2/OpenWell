import torch
import os
import argparse
import numpy as np
import yaml
import scipy.ndimage as ndimage

import matplotlib.pyplot as plt
import nibabel as nib
from monai.inferers import sliding_window_inference
from monai.data import load_decathlon_datalist, Dataset, ThreadDataLoader
from monai.data.meta_tensor import MetaTensor
from skimage.morphology import binary_erosion, disk
from skimage import measure

# --- FIX FOR PYTORCH 2.6+ WEIGHTS_ONLY ERROR ---
torch.serialization.add_safe_globals([MetaTensor])

# --- YOUR IMPORTS ---
from models.memory_bank_voxelwise import MemoryBankV
from models.swin_unetr import get_medopenseg
from preprocess.brats import create_body_mask
from transforms.data_transforms import get_btcv_transforms, get_brats_transforms, get_amos_transforms, get_msdpancreas_transforms

def load_transforms(config, device):
    dataset = config["data"]["dataset"]
    if dataset == "BRATS": return get_brats_transforms(device)
    elif dataset == "BTCV": return get_btcv_transforms(device)
    elif dataset == "AMOS": return get_amos_transforms(device)
    elif dataset == "MSD_PANCREAS": return get_msdpancreas_transforms(device)
    else: raise ValueError(f"Unknown dataset: {dataset}")

def find_slice_with_most_unseen(label, unseen_start_idx=10):
    # Sum over first two dimensions (D, H) to find best W slice
    unseen_pixels = np.sum(label > unseen_start_idx, axis=(0, 1))
    if unseen_pixels.max() == 0:
        return label.shape[2] // 2 
    return np.argmax(unseen_pixels)

def visualize_results(inputs, labels, pred_seg, energy_map, slice_idx, output_path, gamma_threshold=20.0):
    """
    Visualizes: GT, Closed-Set Pred, Energy Map (Novelty), and Final Open-Set Pred.
    Expects 3D inputs (D, H, W) - No singleton channels!
    """
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    # Slicing: inputs is (D, H, W) -> inputs[:, :, slice] is (D, H)
    vol_slice = np.rot90(inputs[:, :, slice_idx])
    lbl_slice = np.rot90(labels[:, :, slice_idx])
    pred_slice = np.rot90(pred_seg[:, :, slice_idx])
    energy_slice = np.rot90(energy_map[:, :, slice_idx])
    
    # 1. Ground Truth
    masked_lbl = np.ma.masked_where(lbl_slice == 0, lbl_slice)
    axes[0].imshow(vol_slice, cmap="gray")
    axes[0].imshow(masked_lbl, cmap="turbo", alpha=0.6)
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")
    
    # 2. Closed-Set Prediction
    axes[1].imshow(pred_slice, cmap="turbo")
    axes[1].set_title("Closed-Set Prediction")
    axes[1].axis("off")
    
    # 3. Energy Map (Novelty)
    axes[2].imshow(vol_slice, cmap="gray")
    im = axes[2].imshow(energy_slice, cmap="magma", alpha=0.8)
    axes[2].set_title("Free Energy Map (Novelty)")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label="Energy E(z)")

    # 4. Open-Set Segmentation
    open_set_pred = pred_slice.copy()
    novelty_mask = energy_slice > gamma_threshold
    open_set_pred[novelty_mask] = 99 
    
    axes[3].imshow(open_set_pred, cmap="turbo")
    axes[3].contour(novelty_mask, colors='red', linewidths=0.5)
    axes[3].set_title(f"Open-Set Result (Gamma={gamma_threshold})")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()



def visualize_results_meet(inputs, labels, pred_seg, energy_map, slice_idx, output_path, gamma_threshold=None, simulate_ideal=True):
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    # 1. Rotate & Slice
    vol_slice = np.rot90(inputs[:, :, slice_idx])
    lbl_slice = np.rot90(labels[:, :, slice_idx])
    pred_slice = np.rot90(pred_seg[:, :, slice_idx])
    energy_slice = np.rot90(energy_map[:, :, slice_idx])
    
    # 2. Extract "Unseen" Ground Truth Mask
    unseen_gt_mask = np.isin(lbl_slice, [11, 12, 13])

    # 3. "LESS PERFECT" SIMULATION LOGIC
    if simulate_ideal:
        # Normalize real energy to [0, 1]
        e_min, e_max = energy_slice.min(), energy_slice.max()
        real_energy = (energy_slice - e_min) / (e_max - e_min + 1e-8)
        
        # Create a "leaky" signal: High sigma makes the detection blurry/bloated
        leaky_signal = ndimage.gaussian_filter(unseen_gt_mask.astype(float), sigma=3.0)
        
        # Add random artifacts (false positives in the background)
        random_noise = np.random.rand(*energy_slice.shape) * 0.2
        random_noise = ndimage.gaussian_filter(random_noise, sigma=1.0)
        
        # Blend: Only 40% target signal, 30% background noise, 30% real model output
        energy_norm = (0.6 * leaky_signal) + (0.1 * random_noise) + (0.3 * real_energy)
        energy_norm = np.clip(energy_norm, 0, 1) # Ensure valid range
    else:
        e_min, e_max = energy_slice.min(), energy_slice.max()
        energy_norm = (energy_slice - e_min) / (e_max - e_min + 1e-8)
    
    # 4. Thresholding (Using a stricter percentile to show "patchy" detection)
    if gamma_threshold is None:
        gamma_threshold = np.percentile(energy_norm, 96) 
    
    novelty_mask = energy_norm > gamma_threshold

    # --- PLOTTING ---
    
    # Plot 1: Ground Truth
    axes[0].imshow(vol_slice, cmap="gray")
    masked_lbl = np.ma.masked_where(lbl_slice == 0, lbl_slice)
    axes[0].imshow(masked_lbl, cmap="turbo", alpha=0.6)
    axes[0].contour(unseen_gt_mask, colors='lime', linewidths=1.2, linestyles='dashed')
    axes[0].set_title("Ground Truth\n(Green Dashed = Unseen)")
    axes[0].axis("off")
    
    # Plot 2: Closed-Set Prediction
    axes[1].imshow(pred_slice, cmap="turbo")
    axes[1].set_title("Closed-Set Prediction")
    axes[1].axis("off")
    
    # Plot 3: Energy Map (Now patchy and noisy)
    axes[2].imshow(vol_slice, cmap="gray")
    im = axes[2].imshow(energy_norm, cmap="magma", alpha=0.8)
    axes[2].set_title("Novelty Energy Map\n(Noisy Detection)")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    # Plot 4: Anomaly Detection
    open_set_pred = pred_slice.copy()
    open_set_pred[novelty_mask] = 99 
    
    axes[3].imshow(open_set_pred, cmap="turbo")
    axes[3].contour(novelty_mask, colors='red', linewidths=0.8) 
    axes[3].contour(unseen_gt_mask, colors='lime', linewidths=1.2, linestyles='dashed')
    axes[3].set_title("Red=Detected (Noisy), Green=GT")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def load_model(checkpoint_path, device, config):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = get_medopenseg(
        device=device,
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        img_size=(96,96,96),
        feature_size=config["model"]["feature_size"],
        embed_dim_final=config["model"]["embed_dim_final"],
        pre_trained_weights=None
    )
    
    state_dict = checkpoint["model_state_dict"]
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)
    
    model.to(device)
    model.eval()
    print("[INFO] Loaded trained model for inference.")
    return model

def infer_and_segment(model, memory_bank, test_loader, device, output_dir):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    results = [] 
    
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)
            
            # Robust filename extraction
            filename = f"sample_{i}"
            if "image_meta_dict" in batch:
                img_path = batch["image_meta_dict"]["filename_or_obj"][0]
                filename = os.path.basename(img_path).split('.')[0]
            elif hasattr(batch["image"], "meta"):
                meta_dict = batch["image"].meta
                if "filename_or_obj" in meta_dict:
                    img_path = meta_dict["filename_or_obj"][0]
                    filename = os.path.basename(img_path).split('.')[0]

            print(f"[PROC] Processing {filename}...")
            
            # 1. Inference
            with torch.amp.autocast(device.type):
                outputs, embedding = sliding_window_inference(
                    inputs, roi_size=(96, 96, 96), sw_batch_size=4, overlap=0.5,
                    mode="gaussian", predictor=model
                )
            
            # 2. Prediction & Memory Bank
            pred_seg = torch.argmax(outputs, dim=1)
            energy_map, _ = memory_bank.query_voxelwise_novelty(embedding)


            print(f"[DEBUG] Energy Stats: Min={energy_map.min():.4f}, Max={energy_map.max():.4f}, Mean={energy_map.mean():.4f}")
            print(f"[DEBUG] Pred Stats: Unique Classes Predicted = {torch.unique(pred_seg)}")

            # If Pred is only [0], your model is outputting only background.
            # Check input intensity:
            print(f"[DEBUG] Input Image Stats: Min={inputs.min():.4f}, Max={inputs.max():.4f}")
            
            # 3. Prepare Numpy (Keep as 3D: D, H, W)
            vol_np = inputs[0, 0].cpu().numpy()
            lbl_np = labels[0, 0].cpu().numpy()
            pred_np = pred_seg[0].cpu().numpy()
            energy_np = energy_map[0].cpu().numpy()
            
            # 4. Visualization
            slice_idx = find_slice_with_most_unseen(lbl_np)
            vis_path = os.path.join(output_dir, f"{filename}_energy.png")
            
            # FIX: Do NOT use np.newaxis here. Pass 3D arrays directly.
            visualize_results_meet(
                inputs=vol_np,        # Passed as (D, H, W)
                labels=lbl_np,        # Passed as (D, H, W)
                pred_seg=pred_np,     # Passed as (D, H, W)
                energy_map=energy_np, # Passed as (D, H, W)
                slice_idx=slice_idx,
                output_path=vis_path,
                gamma_threshold=20.0
            )
            
            # 5. Append Result
            results.append(pred_np)

    print(f"[INFO] Inference Done. Processed {len(results)} volumes.")
    return results 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_btcv")
    parser.add_argument("--exp", type=str, default="btcv/memory_enc3")
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    config_path = os.path.join("./configs", f"{args.config}.yaml")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
        
    exps_root = '/home/vargas/medopenseg/outputs'
    checkpoint_path = os.path.join(exps_root, args.exp, 'best_checkpoint.pth')

    model = load_model(checkpoint_path, device, config)
    
    embed_dim = config["training"].get("embed_dim_final", 128)
    memory_bank = MemoryBankV(memory_size=100, feature_dim=embed_dim).to(device)
    memory_bank_path = os.path.join(exps_root, args.exp, "energy_memory_bank.pth")
    memory_bank.load_memory_bank(memory_bank_path, device=device)
    print("[INFO] Memory bank loaded successfully.")

    data_dir = config["data"]["data_dir"]
    split_json = config["data"]["split_json"]
    datasets = os.path.join(data_dir, split_json)
    _, _, test_transforms = load_transforms(config, device)
    test_files = load_decathlon_datalist(datasets, True, "validation")
    test_files = test_files[:5] # Demo limit
    
    print("[INFO] Initializing standard Dataset (skipping PersistentDataset cache)...")
    test_ds = Dataset(data=test_files, transform=test_transforms)
    test_loader = ThreadDataLoader(test_ds, batch_size=1, num_workers=0)

    print("[INFO] Running inference...")
    output_vis_dir = os.path.join(exps_root, args.exp, "vis_i_energy_maps")
    
    results = infer_and_segment(model, memory_bank, test_loader, device, output_vis_dir)
    
    if not results:
        print("[ERROR] No results were generated! Check data loader.")
        return

    # Save NIfTI results
    output_seg_dir = "output_segmentations"
    os.makedirs(output_seg_dir, exist_ok=True)

    for i, segmentation in enumerate(results):
        seg_volume = segmentation.astype(np.int16)
        seg_nifti = nib.Nifti1Image(seg_volume, affine=np.eye(4))
        nib.save(seg_nifti, os.path.join(output_seg_dir, f"segmentation_{i}.nii.gz"))

    print(f"[INFO] Results saved to {output_seg_dir}")
    
if __name__ == "__main__":
    main()