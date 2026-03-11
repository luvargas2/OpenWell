import torch
import matplotlib.pyplot as plt
import numpy as np
import os

class TrainingVisualizer:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def log_snapshot(self, inputs, labels, preds, raw_energy, prob_bg, epoch, step):
        """
        Saves a visual snapshot of the current training state.
        inputs: [B, C, D, H, W]
        labels: [B, 1, D, H, W]
        preds:  [B, D, H, W]
        raw_energy: [B, D, H, W] (from Memory Bank, BEFORE the gate)
        prob_bg: [B, D, H, W] (Softmax probability of class 0)
        """
        img = inputs[0, 0].detach().cpu().numpy()
        lbl = labels[0, 0].detach().cpu().numpy()
        prd = preds[0].detach().cpu().numpy()
        
        nrg_vol = raw_energy[0].detach().cpu().numpy()
        pbg_vol = prob_bg[0].detach().cpu().numpy()
        
        # --- Volume-Level Normalization for Raw Energy ---
        # We use quantiles to ignore crazy outliers that ruin the color map
        q_low = np.quantile(nrg_vol, 0.05)
        q_high = np.quantile(nrg_vol, 0.99)
        nrg_vol_norm = np.clip((nrg_vol - q_low) / (q_high - q_low + 1e-6), 0, 1)

        # ---------------------------------------------------------
        # THE FIX: Physical Body Mask
        # Bypass the network's overconfident softmax gate.
        # We create a mask dynamically from the CT intensities to silence the air.
        # ---------------------------------------------------------
        air_threshold = np.min(img) + 0.1 * (np.max(img) - np.min(img))
        body_mask = (img > air_threshold).astype(np.float32) 

        # The Final Novelty is now: Thermodynamic Energy * Physical Body Mask
        final_nrg_vol = nrg_vol_norm * body_mask
        
        # We still calculate the network's gate purely for visualization in Panel 4
        gate_vol = 1.0 - pbg_vol

        # --- Smart Slice Selection ---
        ignore_counts = np.sum(lbl == 255, axis=(1, 2))
        max_ignore = np.max(ignore_counts)
        
        if max_ignore > 0:
            slice_idx = np.argmax(ignore_counts)
        else:
            organ_counts = np.sum((lbl > 0) & (lbl < 255), axis=(1, 2))
            slice_idx = np.argmax(organ_counts) if np.max(organ_counts) > 0 else img.shape[0] // 2

        # Extract Slices
        img_slice = img[slice_idx]
        lbl_slice = lbl[slice_idx]
        prd_slice = prd[slice_idx]
        nrg_slice = nrg_vol_norm[slice_idx]    # Raw Energy
        gate_slice = gate_vol[slice_idx]       # The Network's Gate
        final_slice = final_nrg_vol[slice_idx] # The Physically Gated Energy

        # --- Plotting (Expanded to 6 panels to see the TRUTH) ---
        fig, ax = plt.subplots(1, 6, figsize=(28, 5))
        
        # A. Input
        ax[0].imshow(np.rot90(img_slice), cmap="gray")
        ax[0].set_title(f"Ep {epoch} Input")
        
        # B. Ground Truth
        masked_lbl = np.ma.masked_where((lbl_slice == 0) | (lbl_slice == 255), lbl_slice)
        ax[1].imshow(np.rot90(img_slice), cmap="gray")
        ax[1].imshow(np.rot90(masked_lbl), cmap="turbo", alpha=0.5, vmin=0, vmax=14)
        if (lbl_slice == 255).any():
            ax[1].contour(np.rot90(lbl_slice == 255), colors='lime', linewidths=1.5)
        ax[1].set_title("GT (Green=Unseen)")

        # C. Prediction
        ax[2].imshow(np.rot90(prd_slice), cmap="turbo", vmin=0, vmax=14)
        ax[2].set_title("Prediction")

        # D. Gate (1 - P_bg)
        im3 = ax[3].imshow(np.rot90(gate_slice), cmap="gray", vmin=0, vmax=1)
        ax[3].set_title("Network Gate (1 - P_bg)\nWhite=Foreground")
        plt.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.04)

        # E. Raw Energy
        im4 = ax[4].imshow(np.rot90(nrg_slice), cmap="magma", vmin=0, vmax=1)
        ax[4].set_title("Raw Free Energy\n(Memory Bank Only)")
        plt.colorbar(im4, ax=ax[4], fraction=0.046, pad=0.04)

        # F. Final Gated Score (Using Physical Body Mask)
        im5 = ax[5].imshow(np.rot90(final_slice), cmap="magma", vmin=0, vmax=1)
        ax[5].set_title("Final Novelty Map\n(Energy * Body Mask)")
        plt.colorbar(im5, ax=ax[5], fraction=0.046, pad=0.04)

        for a in ax: a.axis('off')

        save_path = os.path.join(self.save_dir, f"epoch_{epoch:03d}_step_{step:04d}.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        return save_path
    