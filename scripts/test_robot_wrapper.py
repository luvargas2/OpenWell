import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # openwell/
sys.path.insert(0, str(ROOT))

from src.interactive.robot_users_omnimedseg.simulate_clicks_3d import (
    EDTBackend, ComponentSelector, ClickPlacer, ClickSimulator
)

class OpenWellEnroller:
    def __init__(self, energy_threshold=0.8):
        """
        energy_threshold: The Free Energy value above which we consider a voxel "Unseen"
        """
        self.energy_threshold = energy_threshold
        
        # Initialize Zdravko's Robot pipeline
        self.edt_backend = EDTBackend()
        self.selector = ComponentSelector(connectivity=26)
        self.placer = ClickPlacer(self.edt_backend)
        self.robot = ClickSimulator(self.selector, self.placer)

    def get_clicks_from_energy(self, raw_energy_map: torch.Tensor):
        """
        Takes the continuous energy map from Memory Bank, thresholds it,
        and uses the Robot to find the optimal click coordinate.
        
        raw_energy_map: [B, D, H, W] tensor (normalized 0 to 1)
        """
        # We process the first item in the batch
        energy_3d = raw_energy_map[0].detach()
        
        # 1. Threshold the Energy Map to create a Binary Anomaly Mask
        anomaly_mask = (energy_3d > self.energy_threshold)
        
        # Check if any anomaly was detected
        if not anomaly_mask.any():
            print("[INFO] No highly novel structures detected.")
            return None, anomaly_mask
            
        # 2. Ask Zdravko's Robot to click the center of the largest unseen anomaly
        try:
            click_coord = self.robot.simulate(
                mask=anomaly_mask,
                comp_strategy="largest",  
                click_strategy="center"   
            )
            print(f"[SUCCESS] Robot clicked at 3D coordinate (Z, Y, X): {click_coord}")
            return click_coord, anomaly_mask
            
        except ValueError as e:
            print(f"[WARNING] Robot failed to place click: {e}")
            return None, anomaly_mask


def test_interactive_pipeline():
    print("--- Testing OpenWell Interactive Robot ---")
    
    # 1. Create a fake 3D Energy Map [B, D, H, W] representing a shape of (1, 64, 128, 128)
    # Background energy is low (e.g., ~0.1)
    fake_energy = torch.ones((1, 64, 128, 128), dtype=torch.float32) * 0.1
    
    # 2. Inject a "Tumor" (High Energy Blob) into the volume
    # Let's put it around Z=32, Y=80, X=40
    z_c, y_c, x_c = 32, 80, 40
    radius = 12
    
    # Create a simple spherical gradient for the energy
    grid_z, grid_y, grid_x = torch.meshgrid(
        torch.arange(64), torch.arange(128), torch.arange(128), indexing='ij'
    )
    dist = torch.sqrt((grid_z - z_c)**2 + (grid_y - y_c)**2 + (grid_x - x_c)**2)
    
    # Inside the radius, energy goes up to 0.95
    tumor_mask = dist <= radius
    fake_energy[0][tumor_mask] = 0.95 - (dist[tumor_mask] / radius) * 0.1 

    # 3. Initialize our Wrapper and Run it
    enroller = OpenWellEnroller(energy_threshold=0.8)
    click_coord, binary_mask = enroller.get_clicks_from_energy(fake_energy)
    
    # 4. Visualization to prove it worked
    if click_coord:
        z_click, y_click, x_click = click_coord

        # ... (Add this inside test_interactive_pipeline() after you get click_coord) ...

    if click_coord:
        z_click, y_click, x_click = click_coord
        
        print("[INFO] Simulating FastGeodis Expansion...")
        from src.interactive.geodesic import GeodesicSegmenter
        
        # Create a fake CT image (just zeros for this test)
        fake_image = torch.zeros((1, 1, 64, 128, 128), dtype=torch.float32)
        fake_energy_5d = fake_energy.unsqueeze(1) # [1, 1, D, H, W]
        
        segmenter = GeodesicSegmenter(distance_threshold=0.2)
        new_mask, geo_dist = segmenter.compute_pre_segmentation(fake_image, fake_energy_5d, click_coord)
        
        # Plot the resulting 3D mask grown from the click
        mask_slice = new_mask[0, 0, z_click, :, :].cpu().numpy()
        dist_slice = geo_dist[0, 0, z_click, :, :].cpu().numpy()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(dist_slice, cmap="viridis")
        axes[0].scatter(x_click, y_click, color='red', marker='x', s=100)
        axes[0].set_title("Geodesic Distance Map (Darker = Closer)")
        
        axes[1].imshow(mask_slice, cmap="gray")
        axes[1].set_title("Final Enrolled Tumor Mask")
        
        plt.tight_layout()
        plt.savefig("geodesic_test_output.png")
        print("[INFO] Saved Geodesic visualization to 'geodesic_test_output.png'")
        
        # Extract the 2D slices at the Z-level where the click occurred
        energy_slice = fake_energy[0, z_click, :, :].numpy()
        mask_slice = binary_mask[z_click, :, :].cpu().numpy()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot 1: The Raw Energy Map
        axes[0].imshow(energy_slice, cmap="magma", vmin=0, vmax=1)
        axes[0].scatter(x_click, y_click, color='cyan', marker='*', s=200, label='Robot Click')
        axes[0].set_title(f"Simulated Raw Energy Map (Slice Z={z_click})")
        axes[0].legend()
        
        # Plot 2: The Binary Mask Zdravko's Robot saw
        axes[1].imshow(mask_slice, cmap="gray")
        axes[1].scatter(x_click, y_click, color='red', marker='x', s=150, label='EDT Center')
        axes[1].set_title("Thresholded Anomaly Mask & Click")
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig("robot_test_output.png")
        print("[INFO] Saved visualization to 'robot_test_output.png'")

if __name__ == "__main__":
    test_interactive_pipeline()