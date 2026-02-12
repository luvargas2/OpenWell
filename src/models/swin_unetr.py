# src/models/swin_unetr.py

from __future__ import annotations
import torch
import torch.nn as nn
from monai.networks.nets import SwinUNETR
from monai.utils import ensure_tuple_rep

class MedOpenSeg(SwinUNETR):
    """
    Extension of MONAI's SwinUNETR that returns both Segmentation Logits 
    AND the Bottleneck Embeddings for the Memory Bank.
    """
    def __init__(
        self,
        img_size,
        in_channels,
        out_channels,
        feature_size=48,
        embed_dim_final=128,
        use_checkpoint=True,
        spatial_dims=3,
        **kwargs # Pass through other SwinUNETR args
    ):
        super().__init__(
            img_size=img_size,
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            **kwargs
        )
        
        # Projection head to map high-dim features to the Memory Bank dimension
        # SwinUNETR's bottleneck size is usually 16 * feature_size (e.g. 16*48=768)
        # But we take the output of the decoder's last stage (feature_size)
        self.embed_out = nn.Conv3d(feature_size, embed_dim_final, kernel_size=1)

    def forward(self, x_in):
        # We override forward to access intermediate features
        # Note: This relies on the internal structure of MONAI's SwinUNETR.
        # Ideally, we use the standard forward but capture the second-to-last output.
        # Since MONAI's forward() returns only logits, we replicate the decoder flow here.
        
        # 1. Encoder (Swin ViT)
        hidden_states_out = self.swinViT(x_in, self.normalize)
        
        # 2. Encoder Projections (UNETR style)
        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])
        
        # 3. Decoder
        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        
        # 4. Final Layers
        out = self.decoder1(dec0, enc0) # [B, feature_size, H, W, D]
        
        # A. Segmentation Logits
        logits = self.out(out)
        
        # B. Embeddings for Memory Bank
        embedding = self.embed_out(out)
        
        return logits, embedding

    def load_from(self, weights):
        """Loads weights, ignoring shape mismatches for the head."""
        with torch.no_grad():
            model_state = self.state_dict()
            pretrained_state = weights if "state_dict" not in weights else weights["state_dict"]
            
            # Filter out keys that don't match (like the final head if classes differ)
            pretrained_state = {
                k: v for k, v in pretrained_state.items() 
                if k in model_state and v.shape == model_state[k].shape
            }
            
            self.load_state_dict(pretrained_state, strict=False)
            print(f"[INFO] Loaded {len(pretrained_state)}/{len(model_state)} layers from pretrained weights.")

def get_medopenseg(
    device, 
    in_channels, 
    out_channels, 
    img_size=(96, 96, 96), 
    feature_size=48, 
    embed_dim_final=128, 
    pre_trained_weights=None
):
    model = MedOpenSeg(
        in_channels=in_channels,
        img_size=img_size,
        out_channels=out_channels,
        feature_size=feature_size,
        embed_dim_final=embed_dim_final,
        use_checkpoint=True,
    ).to(device)

    if pre_trained_weights:
        try:
            weights = torch.load(pre_trained_weights, map_location=device)
            model.load_from(weights)
            print("[INFO] Pre-trained weights loaded successfully.")
        except Exception as e:
            print(f"[WARNING] Could not load weights: {e}")

    return model