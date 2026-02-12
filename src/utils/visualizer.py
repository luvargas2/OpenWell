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
        inputs: [B, C, D, H, W]
        labels: [B, 1, D, H, W]
        preds: [B, D, H, W] (Argmaxed)
        energy: [B, D, H, W]
        """
        # Take the first item in the batch
        img = inputs[0, 0, :, :, :].detach().cpu().numpy()
        lbl = labels[0, 0, :, :, :].detach().cpu().numpy()
        prd = preds[0, :, :, :].detach().cpu().numpy()
        nrg = energy[0, :, :, :].detach().cpu().numpy()

        # Find interesting slice (center or max label)
        slice_idx = img.shape[0] // 2
        
        fig, ax = plt.subplots(1, 4, figsize=(20, 5))
        
        # 1. Input
        ax[0].imshow(np.rot90(img[slice_idx]), cmap="gray")
        ax[0].set_title(f"Epoch {epoch} Input")
        
        # 2. GT (Show Unseen/Ignored as Green)
        masked_lbl = np.ma.masked_where(lbl == 0, lbl)
        ax[1].imshow(np.rot90(img[slice_idx]), cmap="gray")
        ax[1].imshow(np.rot90(masked_lbl), cmap="turbo", alpha=0.5)
        # Highlight Ignore Index (255)
        ignore_mask = (lbl == 255)
        if ignore_mask.any():
            ax[1].contour(np.rot90(ignore_mask[slice_idx]), colors='lime', linewidths=1)
        ax[1].set_title("GT (Green=Ignore/Unseen)")

        # 3. Prediction
        ax[2].imshow(np.rot90(prd[slice_idx]), cmap="turbo")
        ax[2].set_title("Current Prediction")

        # 4. Energy (The most important check)
        # Normalize for visibility
        n_min, n_max = nrg.min(), nrg.max()
        nrg_norm = (nrg - n_min) / (n_max - n_min + 1e-6)
        im = ax[3].imshow(np.rot90(nrg_norm[slice_idx]), cmap="magma")
        ax[3].set_title(f"Energy [{n_min:.1f}, {n_max:.1f}]")
        plt.colorbar(im, ax=ax[3])

        plt.savefig(os.path.join(self.save_dir, f"epoch_{epoch:04d}_step_{step}.png"))
        plt.close()