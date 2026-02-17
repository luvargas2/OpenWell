import torch
import matplotlib.pyplot as plt
import numpy as np
import os

class TrainingVisualizer:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

    def log_snapshot(self, inputs, labels, preds, energy, epoch, step):
        """
        Saves a visual snapshot of the current training state.
        Automatically selects the slice with the most 'Unseen' (255) content.
        
        inputs: [B, C, D, H, W]
        labels: [B, 1, D, H, W]
        preds: [B, D, H, W]
        energy: [B, D, H, W]
        """
        # 1. Unwrap Batch & CPU (Take first item in batch)
        # inputs is [B, 1, D, H, W] -> img is [D, H, W]
        img = inputs[0, 0, :, :, :].detach().cpu().numpy()
        
        # labels is [B, 1, D, H, W] -> lbl is [D, H, W]
        lbl = labels[0, 0, :, :, :].detach().cpu().numpy()
        
        # preds is [B, D, H, W] -> prd is [D, H, W]
        prd = preds[0, :, :, :].detach().cpu().numpy()
        
        # energy is [B, D, H, W] -> nrg is [D, H, W]
        nrg = energy[0, :, :, :].detach().cpu().numpy()

        # 2. Smart Slice Selection
        # Strategy A: Find slice with most 'Ignored' (255) pixels
        ignore_counts = np.sum(lbl == 255, axis=(1, 2))
        max_ignore = np.max(ignore_counts)
        
        if max_ignore > 0:
            slice_idx = np.argmax(ignore_counts)
            tag = "Focus: Unseen"
        else:
            # Strategy B: If no Unseen, find slice with most Organ pixels (>0)
            organ_counts = np.sum(lbl > 0, axis=(1, 2))
            max_organ = np.max(organ_counts)
            if max_organ > 0:
                slice_idx = np.argmax(organ_counts)
                tag = "Focus: Organ"
            else:
                # Strategy C: Fallback to Middle
                slice_idx = img.shape[0] // 2
                tag = "Focus: Mid"

        # 3. Extract the chosen 2D slice
        img_slice = img[slice_idx, :, :]
        lbl_slice = lbl[slice_idx, :, :]
        prd_slice = prd[slice_idx, :, :]
        nrg_slice = nrg[slice_idx, :, :]

        # 4. Plotting
        fig, ax = plt.subplots(1, 4, figsize=(20, 5))
        
        # A. Input Image
        ax[0].imshow(np.rot90(img_slice), cmap="gray")
        ax[0].set_title(f"Ep {epoch} Input")
        ax[0].axis('off')
        
        # B. Ground Truth (Green = Unseen/255)
        # Create a mask for overlay
        masked_lbl = np.ma.masked_where(lbl_slice == 0, lbl_slice)
        ax[1].imshow(np.rot90(img_slice), cmap="gray")
        ax[1].imshow(np.rot90(masked_lbl), cmap="turbo", alpha=0.5, vmin=0, vmax=14)
        
        # Contour for Ignore Index (255)
        ignore_mask = (lbl_slice == 255)
        if ignore_mask.any():
            ax[1].contour(np.rot90(ignore_mask), colors='lime', linewidths=1.5)
        ax[1].set_title("GT (Green=Ignored)")
        ax[1].axis('off')

        # C. Prediction
        ax[2].imshow(np.rot90(prd_slice), cmap="turbo", vmin=0, vmax=14)
        ax[2].set_title("Prediction")
        ax[2].axis('off')

        # D. Energy Map
        n_min, n_max = nrg_slice.min(), nrg_slice.max()
        # Normalize for visibility: (x - min) / (max - min)
        if n_max - n_min > 1e-6:
            nrg_norm = (nrg_slice - n_min) / (n_max - n_min)
        else:
            nrg_norm = nrg_slice
            
        im = ax[3].imshow(np.rot90(nrg_norm), cmap="magma")
        ax[3].set_title(f"Energy [{n_min:.1f}, {n_max:.1f}]")
        ax[3].axis('off')
        plt.colorbar(im, ax=ax[3], fraction=0.046, pad=0.04)

        # 5. Save and Return Path
        save_path = os.path.join(self.save_dir, f"epoch_{epoch:03d}_step_{step:04d}.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        return save_path