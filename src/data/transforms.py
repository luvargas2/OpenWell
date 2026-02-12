import torch
import numpy as np
from monai.transforms import MapTransform
from monai.transforms import (
    LoadImaged, ScaleIntensityRanged, CropForegroundd, Orientationd,
    Spacingd, EnsureTyped, RandCropByPosNegLabeld, RandFlipd, RandRotate90d, 
    RandShiftIntensityd, Compose, NormalizeIntensityd, RandScaleIntensityd, 
    ToTensord, Lambdad, Resized, RandSpatialCropd
)

# --- GLOBAL CONFIG ---
IGNORE_LABEL = 255  # The 'Void' class.

def map_unseen_to_ignore(label_data, unseen_ids):
    """
    Maps specific class IDs to IGNORE_LABEL (255).
    Theory: This prevents the model from learning 'Unseen = Background'.
    """
    if isinstance(label_data, torch.Tensor):
        out = label_data.clone()
    else:
        out = label_data.copy()
        
    for uid in unseen_ids:
        out[out == uid] = IGNORE_LABEL
    return out

class CombineTumorLabels(MapTransform):
    """
    A transform class to combine all tumor sub-regions into a single label.

    This transform modifies the `label` key in the input dictionary.
    All non-zero labels are converted to `1`, representing a unified tumor region.

    Args:
        data (dict): Input dictionary with a "label" key, containing the label array.

    Returns:
        dict: Updated dictionary with combined tumor labels.
    """
    def __call__(self, data):
        data["label"][data["label"] > 0] = 1  # Combine all tumor regions
        return data
    

def get_transforms(dataset_name, device, num_samples=4):
    """
    Factory function to get transforms based on dataset name.
    """
    if dataset_name == "BTCV":
        return get_btcv_transforms(device, num_samples)
    elif dataset_name == "AMOS":
        return get_amos_transforms(device, num_samples)
    elif dataset_name == "MSD_PANCREAS":
        return get_msdpancreas_transforms(device, num_samples)
    elif dataset_name == "BRATS":
        return get_brats_transforms(device, num_samples)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def get_btcv_transforms(device, num_samples=4):
    # BTCV Configuration: Pancreas(11), Adrenal(12, 13) are Unseen
    unseen_ids = [11, 12, 13]

    train_transforms = Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        Lambdad(keys="label", func=lambda x: map_unseen_to_ignore(x, unseen_ids)),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        EnsureTyped(keys=["image", "label"], device=device, track_meta=False),
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
        Lambdad(keys="label", func=lambda x: map_unseen_to_ignore(x, unseen_ids)),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
    ])
    
    # TEST TRANSFORMS: Do NOT mask unseen classes (we need them for ground truth evaluation)
    test_transforms = Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
    ])

    return train_transforms, val_transforms, test_transforms

def get_amos_transforms(device, num_samples=4):
    # AMOS Unseen: 
    unseen_ids = [11, 12, 13, 14, 15] 

    train_transforms = Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
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
        ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
    ])    
    return train_transforms, val_transforms, test_transforms

def get_msdpancreas_transforms(device, num_samples=4):
    unseen_ids = [2] # Assuming 2=Tumor

    train_transforms = Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
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

def get_brats_transforms(device, num_samples=4):
    
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
            # CombineTumorLabels(keys="label"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ToTensord(keys=["image", "label"]),
    ])
    
    test_transforms = Compose([
            LoadImaged(keys=["image", "label"], ensure_channel_first=True),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            ToTensord(keys=["image", "label"]),
    ])
    
    return train_transforms, val_transforms, test_transforms