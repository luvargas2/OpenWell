import torch
import numpy as np

# Import Zdravko's classes
from src.interactive.robot_users_omnimedseg.simulate_clicks_3d import (
    EDTBackend, ComponentSelector, ClickPlacer, ClickSimulator
)

class OpenWellEnroller:
    def __init__(self, energy_threshold=0.8):
        """
        energy_threshold: The Free Energy value above which we consider a voxel "Unseen"
        """
        self.energy_threshold = energy_threshold
        
        # Initialize User's Robot pipeline
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
        # Anything glowing brightly (e.g., > 0.8) becomes a 1.
        anomaly_mask = (energy_3d > self.energy_threshold)
        
        # Check if any anomaly was detected
        if not anomaly_mask.any():
            print("[INFO] No highly novel structures detected.")
            return None
            
        # 2. Ask User's Robot to click the center of the largest unseen anomaly
        try:
            click_coord = self.robot.simulate(
                mask=anomaly_mask,
                comp_strategy="largest",  # Focus on the biggest glowing blob
                click_strategy="center"   # Click the thickest part (argmax of EDT)
            )
            print(f"[INFO] Robot clicked at coordinate: {click_coord}")
            return click_coord
            
        except ValueError as e:
            print(f"[WARNING] Robot failed to place click: {e}")
            return None