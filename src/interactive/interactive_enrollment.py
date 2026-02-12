import torch
import torch.nn.functional as F
import numpy as np

class InteractiveEnrollment:
    """
    Handles the One-Shot adaptation phase.
    """
    def __init__(self, memory_bank, device="cuda"):
        self.memory_bank = memory_bank
        self.device = device

    def generate_guidance_map(self, click_coords, image_shape, sigma=5.0):
        """
        Generates a Gaussian heatmap centered at the click.
        Args:
            click_coords: (z, y, x)
            image_shape: (D, H, W)
            sigma: Standard deviation (spread of influence)
        """
        D, H, W = image_shape
        cz, cy, cx = click_coords
        
        z = torch.arange(D, device=self.device).float()
        y = torch.arange(H, device=self.device).float()
        x = torch.arange(W, device=self.device).float()
        
        # Create coordinate grids
        grid_z, grid_y, grid_x = torch.meshgrid(z, y, x, indexing='ij')
        
        # Compute squared Euclidean distance from click
        dist_sq = (grid_z - cz)**2 + (grid_y - cy)**2 + (grid_x - cx)**2
        
        # Gaussian function: exp(-dist^2 / (2*sigma^2))
        guidance = torch.exp(-dist_sq / (2 * sigma**2))
        
        # Normalize to [0, 1] for weighting
        guidance = guidance / guidance.max()
        
        return guidance

    def enroll_new_class(self, embedding_volume, click_coords, new_class_id):
        """
        Performs the 'Instant Enrollment' of a new energy well.
        Args:
            embedding_volume: [F, D, H, W] tensor (the cached features)
            click_coords: (z, y, x) tuple
            new_class_id: int ID for the new class
        """
        # 1. Generate Uncertainty-Aware Guidance (The Gaussian)
        # In a real app, 'sigma' could be dynamic based on image gradients (Marinov logic)
        guidance_map = self.generate_guidance_map(
            click_coords, 
            embedding_volume.shape[1:], 
            sigma=3.0 # A heuristic for now - ask Zdravko about this!
        )
        
        # 2. Weighted Feature Aggregation
        # Flatten for dot product
        F_dim = embedding_volume.shape[0]
        feats_flat = embedding_volume.view(F_dim, -1) # [F, N]
        weights_flat = guidance_map.view(-1)          # [N]
        
        # Filter weights < threshold to save compute/noise
        mask = weights_flat > 0.01
        valid_feats = feats_flat[:, mask]
        valid_weights = weights_flat[mask]
        
        # Weighted Average Mean Calculation
        # mu_new = sum(w_i * f_i) / sum(w_i)
        weighted_sum = (valid_feats * valid_weights).sum(dim=1)
        mu_new = weighted_sum / valid_weights.sum()
        
        # Normalize to sphere
        mu_new = F.normalize(mu_new, p=2, dim=0)
        
        # 3. Estimate Kappa (Concentration)
        # High variance in features -> Low Kappa (Wide Well)
        # We can use the weighted variance or a fixed high value for "One-Shot" trust.
        # Let's start with a high trust (narrow well) for the specific clicked point.
        kappa_new = torch.tensor(50.0, device=self.device) 
        
        # 4. Update Memory Bank directly
        self.memory_bank.prototypes[new_class_id] = {
            'mu': mu_new,
            'kappa': kappa_new,
            'count': 1 # It's a one-shot sample
        }
        
        print(f"[ENROLLMENT] New Class {new_class_id} enrolled at {click_coords}.")
        return guidance_map # Return for visualization