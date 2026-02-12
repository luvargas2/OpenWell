from monai.transforms import (
    LoadImaged, ScaleIntensityRanged, CropForegroundd, Orientationd,
    Spacingd, EnsureTyped, RandCropByPosNegLabeld, RandFlipd, RandRotate90d, 
    RandShiftIntensityd, Compose, NormalizeIntensityd, RandScaleIntensityd, 
    ToTensord, Resized, Lambdad
)
from preprocess.brats import CombineTumorLabels
import numpy as np
import torch

# --- CONFIGURATION ---
IGNORE_LABEL = 255  # The 'Void' class. Loss function must ignore this index.

def map_unseen_to_ignore(label_data, unseen_ids):
    """
    Maps specific class IDs to the IGNORE_LABEL (255).
    This prevents the model from learning them as 'Background'.
    """
    # Clone to avoid modifying original data if cached
    if isinstance(label_data, torch.Tensor):
        out = label_data.clone()
    else:
        out = label_data.copy()
        
    for uid in unseen_ids:
        # Map Unseen ID -> 255 (Ignore)
        out[out == uid] = IGNORE_LABEL
    return out

def get_btcv_transforms(device, num_samples=4):
    # BTCV Unseen: Pancreas(11), Adrenal(12, 13 - depending on labeling standard)
    # Adjust these IDs if your specific BTCV version differs!
    unseen_ids = [11, 12, 13] 

    train_transforms = Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        
        # --- THE FIX: Explicitly map Unseen to 255 ---
        Lambdad(keys="label", func=lambda x: map_unseen_to_ignore(x, unseen_ids)),
        
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
        
        # Note: We allow cropping centered on 'Ignore' labels (pos=1 includes 255 if treated as fg)
        # This is good! We want the model to see the pixels but generate NO gradients for them.
        RandCropByPosNegLabeld(
            keys=["image", "label"], label_key="label", spatial_size=(96, 96, 96),
            pos=1, neg=1, num_samples=num_samples, image_key="image", image_threshold=0
        ),
        RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.10),
        RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.10),
        RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.10),
        RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50),
    ])

    val_transforms = Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        # Validation should also hide these labels so Dice isn't penalized? 
        # Actually, for Open-Set valid, you might WANT to see them to measure 'Novelty'.
        # But for 'closed set' validation accuracy, we hide them.
        Lambdad(keys="label", func=lambda x: map_unseen_to_ignore(x, unseen_ids)),
        
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
    ])
    
    test_transforms = Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        # DO NOT MASK LABELS IN TEST! We need them for GT evaluation of novelty.
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
    ])

    return train_transforms, val_transforms, test_transforms

def get_amos_transforms(device, num_samples=4):
    # AMOS Unseen: You must define these based on your split!
    # Example placeholder: [14, 15] 
    # Please replace specific IDs here to match 'AmosMapUnseenClasses' logic
    unseen_ids = [10, 11, 12, 13, 14, 15] # Example: Hard organs often excluded

    train_transforms = Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        
        # --- THE FIX ---
        Lambdad(keys="label", func=lambda x: map_unseen_to_ignore(x, unseen_ids)),
        
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
        RandCropByPosNegLabeld(
            keys=["image", "label"], label_key="label", spatial_size=(64, 64, 64),
            pos=1, neg=1, num_samples=num_samples, image_key="image", image_threshold=0
        ),
        RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.10),
        RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.10),
        RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.10),
        RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50),
    ])
    
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        Lambdad(keys="label", func=lambda x: map_unseen_to_ignore(x, unseen_ids)),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
    ])

    test_transforms = Compose([
        LoadImaged(keys=["image"], ensure_channel_first=True, dtype=np.float32),
        LoadImaged(keys=["label"], ensure_channel_first=True, dtype=np.int16),
        # NO MASKING IN TEST
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
    ])    
    return train_transforms, val_transforms, test_transforms

def get_msdpancreas_transforms(device, num_samples=4):
    # MSD Pancreas usually: 0=Bg, 1=Pancreas, 2=Tumor
    # If "Unseen" is Tumor, we map 2 -> 255
    unseen_ids = [2] 

    train_transforms = Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        
        # --- THE FIX ---
        Lambdad(keys="label", func=lambda x: map_unseen_to_ignore(x, unseen_ids)),
        
        ScaleIntensityRanged(keys=["image"], a_min=-87, a_max=199, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
        RandCropByPosNegLabeld(
            keys=["image", "label"], label_key="label", spatial_size=(64, 64, 64),
            pos=1, neg=1, num_samples=num_samples, image_key="image", image_threshold=0
        ),
        RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.10),
        RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.10),
        RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=0.10),
        RandRotate90d(keys=["image", "label"], prob=0.10, max_k=3),
        RandShiftIntensityd(keys=["image"], offsets=0.10, prob=0.50),
    ])
    
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        Lambdad(keys="label", func=lambda x: map_unseen_to_ignore(x, unseen_ids)),
        ScaleIntensityRanged(keys=["image"], a_min=-87, a_max=199, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
    ])
    
    test_transforms = Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        ScaleIntensityRanged(keys=["image"], a_min=-87, a_max=199, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
    ])

    return train_transforms, val_transforms, test_transforms

# BRATS usually requires specialized handling, leaving as is unless you have specific Unseen IDs for it.
def get_brats_transforms(device, num_samples=4):
    # ... (Keep existing implementation if BRATS logic is handled by CombineTumorLabels) ...
    # If BRATS has unseen classes, apply the same 'map_unseen_to_ignore' pattern.
    
    train_transforms = Compose([
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            CombineTumorLabels(keys="label"),
            CropForegroundd(keys=["image", "label"], source_key="image", k_divisible=[96,96,96]),
            RandSpatialCropd(keys=["image", "label"], roi_size=[96,96,96], random_size=False),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            ToTensord(keys=["image", "label"]),
    ])

    val_transforms = Compose([
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            CombineTumorLabels(keys="label"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ToTensord(keys=["image", "label"]),
    ])
    
    test_transforms = Compose([
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ToTensord(keys=["image", "label"]),
    ])
    
    return train_transforms, val_transforms, test_transforms