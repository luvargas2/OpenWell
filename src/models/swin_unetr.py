import torch
from monai.networks.nets import SwinUNETR
from models.swin_medopenseg import MedOpenSeg

def get_swin_unetr_model(device, in_channels=1, out_channels=14,img_size=(96, 96, 96),feature_size=48, pre_trained_weights=None):
    model = SwinUNETR(
        in_channels=in_channels,
        img_size=img_size,
        out_channels=out_channels,
        feature_size=feature_size,
        use_checkpoint=True,
    ).to(device)

    if pre_trained_weights:
        weights = torch.load(pre_trained_weights,weights_only=True)
        model.load_from(weights=weights)
        print("Loaded pre-trained weights!")

    return model

def get_medopenseg(device, in_channels=1, out_channels=14,img_size=(96, 96, 96),feature_size=48,embed_dim_final=128, pre_trained_weights=None):
    model = MedOpenSeg(
        in_channels=in_channels,
        img_size=img_size,
        out_channels=out_channels,
        feature_size=feature_size,
        embed_dim_final=embed_dim_final,
        use_checkpoint=True,
    ).to(device)

    if pre_trained_weights:
        weights = torch.load(pre_trained_weights,weights_only=True)
        model.load_from(weights=weights)
        print("Loaded pre-trained weights!")

    return model


