# OpenWell

**Energy-Based Interactive Discovery for Open-World Medical Segmentation**

OpenWell is a 3D medical image segmentation framework that detects and enrolls anatomical structures unseen during training. It models known anatomy as energy wells on the unit hypersphere using Von Mises-Fisher (vMF) distributions; novel structures manifest as regions of anomalously high free energy, which can then be enrolled with a single user click.

---

## Method Overview

1. **Backbone** — Swin UNETR with a learnable embedding head that maps voxel features onto the unit hypersphere.
2. **Memory Bank** — Stores one vMF prototype (μ, κ) per known class, updated via EMA during training.
3. **Gated Free Energy** — At inference, voxels with energy above an adaptive per-volume threshold are flagged as novel.
4. **Interactive Enrollment** — A single click on a novel region triggers a guided feature averaging step that injects a new class prototype into the memory bank without any re-training.

---

## Repository Structure

```
openwell/
├── configs/                  # YAML experiment configs
│   ├── config_btcv.yaml
│   ├── config_amos.yaml
│   ├── config_brats.yaml
│   └── config_pancreas.yaml
├── scripts/
│   ├── train.py              # Training entry point
│   └── inference.py          # Evaluation + OOD detection
├── src/
│   ├── models/
│   │   ├── swin_unetr.py     # MedOpenSeg model
│   │   └── memory_bank.py    # vMF prototype bank
│   ├── training/
│   │   ├── trainer.py        # Training loop
│   │   └── losses.py         # Dice-CE + vMF loss
│   ├── data/
│   │   ├── loader.py         # DataLoader factory
│   │   └── transforms.py     # MONAI augmentation pipelines
│   └── interactive/
│       └── interactive_enrollment.py  # Click-based enrollment
├── weights/                  # Pre-trained Swin ViT weights
└── outputs/                  # Checkpoints (created at training time)
```

---

## Requirements

- Python ≥ 3.9
- PyTorch with CUDA
- [MONAI](https://monai.io/) ≥ 1.3
- scikit-learn, scipy, nibabel, matplotlib

Install via `uv` (recommended):

```bash
uv sync
```

Or with pip:

```bash
pip install monai torch torchvision scikit-learn scipy nibabel matplotlib
```

Pre-trained Swin ViT weights (`model_swinvit.pt`) should be placed in `weights/`.

---

## Data Preparation

Datasets are expected in [Medical Segmentation Decathlon](http://medicaldecathlon.com/) JSON format. Update `data_dir` and `split_json` in the relevant config file.

| Config | Dataset | Known classes | Unseen (OOD) classes |
|---|---|---|---|
| `config_btcv.yaml` | BTCV | 10 abdominal organs | Pancreas, Adrenal R/L |
| `config_amos.yaml` | AMOS | 10 abdominal organs | 5 additional organs |
| `config_pancreas.yaml` | MSD-Pancreas | Pancreas | — |

---

## Training

```bash
python scripts/train.py --config configs/config_btcv.yaml
```

Checkpoints and the energy memory bank are saved to `checkpoint_dir` defined in the config. Training resumes automatically if `resume: True` and a checkpoint exists.

---

## Inference & OOD Evaluation

```bash
python scripts/inference.py \
    --config config_btcv \
    --exp    exp_name
```
---
