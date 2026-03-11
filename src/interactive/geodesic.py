import torch
import FastGeodis
import numpy as np

class GeodesicSegmenter:
    def __init__(self, distance_threshold=0.5, iterations=4):
        """
        distance_threshold: Controls how far the region grows. 
        iterations: Number of passes for the FastGeodis algorithm.
        """
        self.distance_threshold = distance_threshold
        self.iterations = iterations

    def compute_pre_segmentation(self, image: torch.Tensor, energy_map: torch.Tensor, click_coord: tuple):
        """
        Grows a 3D mask from a single click point using FastGeodis.
        
        image: [B, 1, D, H, W] - The original CT scan (normalized)
        energy_map: [B, 1, D, H, W] - The continuous novelty map
        click_coord: (Z, Y, X)
        """
        device = image.device
        
        # 1. Prepare the Guide Image (Combining CT features and Energy)
        # We want the growth to respect both physical CT boundaries AND the novelty boundaries.
        # Ensure image and energy_map are the same shape [B, 1, D, H, W]
        if energy_map.ndim == 4:
            energy_map = energy_map.unsqueeze(1)
            
        guide_image = image * 0.5 + energy_map * 0.5 
        
        # 2. Prepare the Seed Mask
        # FastGeodis expects the seed locations to be 0.0, and everywhere else 1.0
        seed_mask = torch.ones_like(image)
        z, y, x = click_coord
        seed_mask[0, 0, z, y, x] = 0.0  # The Robot's Click

        # 3. Compute Geodesic Distance
        # We use spacing [1.0, 1.0, 1.0]. Adjust if your CT voxel spacing is highly anisotropic.
        spacing = [1.0, 1.0, 1.0]
        v = 1e10 # Default for FastGeodis
        
        geodesic_dist = FastGeodis.generalised_geodesic3d(
            guide_image, seed_mask, spacing, v, self.iterations
        )
        
        # 4. Threshold into a Binary Mask
        # Close to the click, distance is near 0. Far away, it approaches 1.0.
        # We keep everything below the threshold as the new organ/tumor.
        # (You may need to invert or normalize geodesic_dist depending on the exact CT scaling)
        geodesic_dist_norm = geodesic_dist / (geodesic_dist.max() + 1e-8)
        
        new_structure_mask = (geodesic_dist_norm < self.distance_threshold).float()
        
        return new_structure_mask, geodesic_dist_norm