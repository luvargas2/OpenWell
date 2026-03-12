import torch
import os
import argparse
import numpy as np
import yaml
import sys
import scipy.ndimage as ndimage

import matplotlib.pyplot as plt
import nibabel as nib
from monai.inferers import sliding_window_inference
from monai.data import load_decathlon_datalist, Dataset, ThreadDataLoader
from monai.data.meta_tensor import MetaTensor
from sklearn.metrics import roc_auc_score, average_precision_score

# --- FIX FOR PYTORCH 2.6+ WEIGHTS_ONLY ERROR ---
torch.serialization.add_safe_globals([MetaTensor])

# Add project root to python path so 'src' is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.memory_bank import MemoryBankV
from src.models.swin_unetr import get_medopenseg
from src.data.transforms import get_btcv_transforms, get_brats_transforms, get_amos_transforms, get_msdpancreas_transforms

# BTCV unseen class IDs in raw label space (test transforms do NOT remap these)
UNSEEN_CLASSES      = [11, 12, 13]
UNSEEN_CLASS_NAMES  = {11: "Pancreas", 12: "Adrenal R.", 13: "Adrenal L."}

# ── Nature-publication colour palette for abdominal organs ──────────────────
# RGBA tuples; alpha=0 = transparent (background).
# Palette is perceptually distinct and prints well in greyscale.
ORGAN_RGBA = {
    0:  (0.00, 0.00, 0.00, 0.00),   # background    — transparent
    1:  (0.78, 0.22, 0.22, 0.65),   # spleen        — crimson
    2:  (0.22, 0.53, 0.78, 0.65),   # R. kidney     — steel blue
    3:  (0.22, 0.68, 0.45, 0.65),   # L. kidney     — emerald
    4:  (0.92, 0.68, 0.20, 0.65),   # gallbladder   — amber
    5:  (0.58, 0.34, 0.68, 0.65),   # esophagus     — violet
    6:  (0.92, 0.52, 0.18, 0.65),   # liver         — warm orange
    7:  (0.20, 0.72, 0.76, 0.65),   # stomach       — teal
    8:  (0.84, 0.15, 0.20, 0.65),   # aorta         — scarlet
    9:  (0.35, 0.58, 0.84, 0.65),   # IVC           — cornflower
    10: (0.70, 0.45, 0.78, 0.65),   # portal veins  — lilac
    99: (0.96, 0.18, 0.56, 0.85),   # enrolled      — vivid pink
}

def _seg_rgba(seg_np):
    """Convert integer label map [H, W] → RGBA [H, W, 4] using ORGAN_RGBA."""
    h, w = seg_np.shape
    out = np.zeros((h, w, 4), dtype=np.float32)
    for lbl, rgba in ORGAN_RGBA.items():
        mask = (seg_np == lbl)
        out[mask] = rgba
    return out


def load_transforms(config, device):
    dataset = config["data"]["dataset"]
    if dataset == "BRATS":   return get_brats_transforms(device)
    elif dataset == "BTCV":  return get_btcv_transforms(device)
    elif dataset == "AMOS":  return get_amos_transforms(device)
    elif dataset == "MSD_PANCREAS": return get_msdpancreas_transforms(device)
    else: raise ValueError(f"Unknown dataset: {dataset}")


def find_slice_with_most_unseen(label):
    """Return the W-axis index with the most unseen-class voxels."""
    mask = np.isin(label, UNSEEN_CLASSES)
    counts = np.sum(mask, axis=(0, 1))   # label is [D, H, W]
    return int(np.argmax(counts)) if counts.max() > 0 else label.shape[2] // 2


# ==========================================
# INTERACTIVE ENROLLMENT FUNCTIONS
# ==========================================
def simulate_user_click(label_np):
    """Simulates a user clicking on the centre of a novel anomaly."""
    mask = np.isin(label_np, UNSEEN_CLASSES)
    if not np.any(mask):
        return None
    coords = ndimage.center_of_mass(mask)
    return tuple(int(c) for c in coords)   # (d, h, w) for [D, H, W] array


def create_guidance_map(shape, click_coord, vol_np=None, sigma=6.0, beta=5.0):
    """
    Gaussian guidance map G centred at click_coord, modulated by image gradient
    magnitude to respect local boundaries (paper Eq. 9):

      G(v) = exp(−||v−p||²/2σ²) · exp(−β·||∇I(v)||/max||∇I||)

    The gradient term suppresses guidance across strong CT boundaries, preventing
    the local feature average from bleeding across organ edges.
    vol_np: [D, H, W] normalised CT intensity (float32).
    """
    z, y, x = np.indices(shape)
    cz, cy, cx = click_coord
    dist_sq = (z - cz)**2 + (y - cy)**2 + (x - cx)**2
    gaussian = np.exp(-dist_sq / (2 * sigma**2))

    if vol_np is not None:
        grad_mag = ndimage.gaussian_gradient_magnitude(vol_np.astype(np.float32), sigma=1.0)
        grad_norm = grad_mag / (grad_mag.max() + 1e-8)
        boundary_suppression = np.exp(-beta * grad_norm)
        heatmap = gaussian * boundary_suppression
        heatmap = heatmap / (heatmap.max() + 1e-8)   # normalise to [0, 1]
    else:
        heatmap = gaussian

    return torch.tensor(heatmap, dtype=torch.float32)


def enroll_new_class(embedding, guidance_map, device):
    """
    Calculates mu_new and kappa_new from the guided local features (Section 3.5).
    """
    import torch.nn.functional as F

    g = guidance_map.to(device).unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]

    weighted_features = embedding * g
    sum_g = g.sum() + 1e-8
    mean_vec = weighted_features.sum(dim=(2, 3, 4)) / sum_g  # [1, C]
    mu_new = F.normalize(mean_vec, p=2, dim=1)

    F_dim = embedding.shape[1]
    R = mean_vec.norm(p=2, dim=1).clamp(min=1e-6, max=0.999)
    kappa_new = (R * F_dim - R**3) / (1 - R**2 + 1e-6)
    # Match training cap (raised from 100 → 500 during training)
    kappa_new = kappa_new.clamp(min=1.0, max=500.0)

    return mu_new.squeeze(0), kappa_new.squeeze(0)


# ==========================================
# QUANTITATIVE OOD METRICS
# ==========================================
def _class_metrics(energy_flat, labels_flat, cls_id, known_mask, lam):
    """Compute binary OOD metrics for a single unseen class vs. all known voxels."""
    cls_mask  = (labels_flat == cls_id)
    if not cls_mask.any():
        return None
    eval_mask = known_mask | cls_mask
    scores    = energy_flat[eval_mask]
    gt        = cls_mask[eval_mask].astype(int)
    if gt.sum() == 0 or (1 - gt).sum() == 0:
        return None
    auroc = roc_auc_score(gt, scores)
    auprc = average_precision_score(gt, scores)
    preds = (scores > lam).astype(int)
    tp = int(((preds == 1) & (gt == 1)).sum())
    fp = int(((preds == 1) & (gt == 0)).sum())
    fn = int(((preds == 0) & (gt == 1)).sum())
    recall    = tp / (tp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    f1        = 2 * recall * precision / (recall + precision + 1e-8)
    return dict(auroc=auroc, auprc=auprc,
                recall_at_lambda=recall, precision_at_lambda=precision,
                f1_at_lambda=f1, tp=tp, fp=fp, fn=fn,
                n_cls=int(cls_mask.sum()))


def compute_novelty_metrics(raw_energy_np, labels_np, adaptive_lambda):
    """
    Returns aggregate + per-class OOD detection metrics.

    Positive  = unseen-class voxels (11 Pancreas, 12 Adrenal-R, 13 Adrenal-L)
    Negative  = known-organ voxels (1–10)
    Background (0) excluded from both sets.
    """
    energy_flat = raw_energy_np.reshape(-1).astype(np.float32)
    labels_flat = labels_np.reshape(-1)

    known_mask  = (labels_flat >= 1) & (~np.isin(labels_flat, UNSEEN_CLASSES))
    unseen_mask = np.isin(labels_flat, UNSEEN_CLASSES)

    if not known_mask.any() or not unseen_mask.any():
        print("[METRICS] WARNING: no unseen voxels found in this volume — skipping metrics.")
        return {}

    eval_mask = known_mask | unseen_mask
    scores    = energy_flat[eval_mask]
    gt_binary = unseen_mask[eval_mask].astype(int)

    auroc = roc_auc_score(gt_binary, scores)
    auprc = average_precision_score(gt_binary, scores)

    preds = (scores > adaptive_lambda).astype(int)
    tp = int(((preds == 1) & (gt_binary == 1)).sum())
    fp = int(((preds == 1) & (gt_binary == 0)).sum())
    fn = int(((preds == 0) & (gt_binary == 1)).sum())
    recall    = tp / (tp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    f1        = 2 * recall * precision / (recall + precision + 1e-8)

    # Per-class breakdown
    per_class = {}
    for cls_id in UNSEEN_CLASSES:
        cm = _class_metrics(energy_flat, labels_flat, cls_id, known_mask, adaptive_lambda)
        if cm is not None:
            per_class[cls_id] = cm

    metrics = dict(
        auroc=auroc, auprc=auprc,
        recall_at_lambda=recall, precision_at_lambda=precision, f1_at_lambda=f1,
        tp=tp, fp=fp, fn=fn,
        n_unseen=int(unseen_mask.sum()), n_known=int(known_mask.sum()),
        per_class=per_class,
    )

    print(f"\n[METRICS] === OOD @ λ={adaptive_lambda:.4f} ===")
    print(f"  Aggregate  AUROC={auroc:.4f}  AUPRC={auprc:.4f}  "
          f"Recall={recall:.4f}  F1={f1:.4f}")
    for cls_id, cm in per_class.items():
        print(f"  {UNSEEN_CLASS_NAMES[cls_id]:12s} AUROC={cm['auroc']:.4f}  "
              f"Recall={cm['recall_at_lambda']:.4f}  n={cm['n_cls']:,}")
    print(f"[METRICS] =========================================\n")

    return metrics


# ==========================================
# PUBLICATION-QUALITY VISUALIZATION (4 panels)
# ==========================================
def visualize_publication(
    inputs, labels, pred_seg, gated_energy, open_pred,
    slice_idx, metrics, output_path
):
    """
    Clean 4-panel figure suitable for journal submission.
    Panels: GT | Closed-Set | Novelty Map | Open-World
    """
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 11,
        'axes.titlesize': 12, 'axes.titleweight': 'bold',
    })
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    vol_s   = np.rot90(inputs[:, :, slice_idx])
    lbl_s   = np.rot90(labels[:, :, slice_idx])
    pred_s  = np.rot90(pred_seg[:, :, slice_idx])
    gated_s = np.rot90(gated_energy[:, :, slice_idx])
    open_s  = np.rot90(np.asarray(open_pred)[:, :, slice_idx])
    unseen_gt = np.isin(lbl_s, UNSEEN_CLASSES)

    # Panel 1: GT
    axes[0].imshow(vol_s, cmap="gray")
    masked_lbl = np.ma.masked_where(lbl_s == 0, lbl_s)
    axes[0].imshow(masked_lbl, cmap="tab20", alpha=0.55, vmin=0, vmax=20)
    axes[0].contour(unseen_gt, colors='#00FF00', linewidths=1.5, linestyles='--')
    axes[0].set_title("Ground Truth")
    axes[0].axis("off")

    # Panel 2: Closed-set segmentation
    axes[1].imshow(vol_s, cmap="gray")
    masked_pred = np.ma.masked_where(pred_s == 0, pred_s)
    axes[1].imshow(masked_pred, cmap="tab20", alpha=0.55, vmin=0, vmax=20)
    axes[1].contour(unseen_gt, colors='#00FF00', linewidths=1.0, linestyles='--')
    axes[1].set_title("Closed-Set Segmentation")
    axes[1].axis("off")

    # Panel 3: Novelty map
    gated_max = float(np.percentile(gated_s[gated_s > 0], 99)) if (gated_s > 0).any() else 0.1
    im = axes[2].imshow(gated_s, cmap="hot", vmin=0, vmax=max(gated_max, 1e-6))
    axes[2].contour(unseen_gt, colors='#00FF00', linewidths=1.5, linestyles='--')
    cbar = plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)
    auroc = metrics.get('auroc', 0) if metrics else 0
    axes[2].set_title(f"Novelty Map (AUROC={auroc:.3f})")
    axes[2].axis("off")

    # Panel 4: Open-world result
    axes[3].imshow(vol_s, cmap="gray")
    masked_open = np.ma.masked_where(open_s == 0, open_s % 20)  # mod 20 for colormap
    axes[3].imshow(masked_open, cmap="tab20", alpha=0.55, vmin=0, vmax=20)
    novel_mask = (open_s == 99)
    if novel_mask.any():
        axes[3].contourf(novel_mask, levels=[0.5, 1.5], colors=['red'], alpha=0.45)
    axes[3].contour(unseen_gt, colors='#00FF00', linewidths=1.0, linestyles='--')
    axes[3].set_title("Open-World Segmentation")
    axes[3].axis("off")

    plt.tight_layout(pad=1.5)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    plt.rcParams.update(plt.rcParamsDefault)
    print(f"[PUB] Publication figure saved to {output_path}")


# ==========================================
# LATEX TABLE — per-class mean ± std
# ==========================================
def generate_latex_table(all_metrics, output_path):
    """
    Writes a LaTeX table: rows = unseen classes, columns = OOD metrics,
    values = mean ± std across all test volumes.
    """
    if not all_metrics:
        return

    keys        = ["auroc", "auprc", "recall_at_lambda", "precision_at_lambda", "f1_at_lambda"]
    col_labels  = ["AUROC", "AUPRC", r"Recall@$\lambda$", r"Prec.@$\lambda$", r"F1@$\lambda$"]

    # Gather per-class values across volumes
    cls_data = {cls_id: {k: [] for k in keys} for cls_id in UNSEEN_CLASSES}
    for m in all_metrics:
        for cls_id in UNSEEN_CLASSES:
            cm = m.get("per_class", {}).get(cls_id, {})
            for k in keys:
                if k in cm:
                    cls_data[cls_id][k].append(cm[k])

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Open-set detection on BTCV (mean\,$\pm$\,std over "
                 + str(len(all_metrics)) + r" test volumes). "
                 r"Unseen classes: Pancreas (class 11), Right Adrenal (12), Left Adrenal (13).}")
    lines.append(r"\label{tab:ood_per_class}")
    lines.append(r"\begin{tabular}{l" + "c" * len(keys) + "}")
    lines.append(r"\toprule")
    lines.append("Structure & " + " & ".join(col_labels) + r" \\")
    lines.append(r"\midrule")

    for cls_id in UNSEEN_CLASSES:
        row = UNSEEN_CLASS_NAMES[cls_id]
        for k in keys:
            vals = cls_data[cls_id][k]
            if vals:
                row += rf" & ${np.mean(vals):.3f}_{{\pm{np.std(vals):.3f}}}$"
            else:
                row += " & —"
        row += r" \\"
        lines.append(row)

    # Aggregate row (all unseen classes pooled)
    lines.append(r"\midrule")
    agg_row = r"\textbf{All unseen}"
    for k in keys:
        all_vals = [v for cls_id in UNSEEN_CLASSES for v in cls_data[cls_id][k]]
        if all_vals:
            agg_row += rf" & $\mathbf{{{np.mean(all_vals):.3f}}}{{\pm{np.std(all_vals):.3f}}}$"
        else:
            agg_row += " & —"
    agg_row += r" \\"
    lines.append(agg_row)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    latex = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(latex)

    print(f"\n[LATEX] Per-class table written to {output_path}")
    print("─" * 60)
    print(latex)
    print("─" * 60)


# ==========================================
# NATURE-STYLE QUALITATIVE FIGURE (3 best)
# ==========================================
def visualize_best_examples(vis_data_list, output_path, n_best=3):
    """
    Produces a publication-quality figure showing the n_best volumes sorted
    by descending AUROC.  Layout: n_best rows × 4 columns.
      Col 0 — CT + GT segmentation (unseen contour)
      Col 1 — Closed-set segmentation
      Col 2 — Free-energy novelty map
      Col 3 — Open-world result after enrollment

    Designed to meet Nature / Nature Medicine figure guidelines:
      • 300 dpi, white background, Arial/Helvetica font
      • Perceptually uniform novelty colormap ('magma')
      • Panel labels (a), (b), (c) in bold
      • Thin 0.5-pt panel borders, no tick marks
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    plt.rcParams.update({
        'font.family':       'sans-serif',
        'font.sans-serif':   ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size':          8,
        'axes.titlesize':     8,
        'axes.titleweight':  'bold',
        'axes.linewidth':     0.5,
        'xtick.bottom':       False,
        'ytick.left':         False,
    })

    # Sort by AUROC
    ranked = sorted(
        [d for d in vis_data_list if d.get('metrics')],
        key=lambda d: d['metrics'].get('auroc', 0),
        reverse=True,
    )[:n_best]

    if not ranked:
        print("[NATURE-VIS] No valid examples to plot.")
        return

    n_rows = len(ranked)
    col_titles = [
        "Ground Truth",
        "Closed-Set Segmentation",
        "Free-Energy Novelty Map",
        "Open-World Segmentation",
    ]
    panel_labels = list("abcdefghijkl")   # (a)–(l) for up to 12 panels
    NOVELTY_CMAP = 'magma'

    fig, axes = plt.subplots(n_rows, 4,
                             figsize=(16, n_rows * 4.0),
                             gridspec_kw={'wspace': 0.04, 'hspace': 0.06})
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row_i, data in enumerate(ranked):
        si   = data['slice_idx']
        vol  = np.rot90(data['vol_np'][:, :, si])
        lbl  = np.rot90(data['lbl_np'][:, :, si])
        pred = np.rot90(data['pred_np'][:, :, si])
        gate = np.rot90(data['gated_np'][:, :, si])
        opn  = np.rot90(np.asarray(data['open_np'])[:, :, si])
        ug   = np.isin(lbl, UNSEEN_CLASSES)
        auroc = data['metrics'].get('auroc', 0)

        # ── Column 0: CT + GT ──────────────────────────────────────────
        ax = axes[row_i, 0]
        ax.imshow(vol, cmap='gray', interpolation='lanczos')
        ax.imshow(_seg_rgba(lbl), interpolation='nearest')
        ax.contour(ug, levels=[0.5], colors=['#00E676'], linewidths=0.8, linestyles='--')
        _panel_border(ax)

        # ── Column 1: Closed-set segmentation ─────────────────────────
        ax = axes[row_i, 1]
        ax.imshow(vol, cmap='gray', interpolation='lanczos')
        ax.imshow(_seg_rgba(pred), interpolation='nearest')
        ax.contour(ug, levels=[0.5], colors=['#00E676'], linewidths=0.8, linestyles='--')
        _panel_border(ax)

        # ── Column 2: Novelty map ──────────────────────────────────────
        ax = axes[row_i, 2]
        ax.imshow(vol, cmap='gray', alpha=0.35, interpolation='lanczos')
        gate_max = float(np.percentile(gate[gate > 0], 99)) if (gate > 0).any() else 0.1
        im = ax.imshow(gate, cmap=NOVELTY_CMAP, vmin=0,
                       vmax=max(gate_max, 1e-6), interpolation='bilinear')
        ax.contour(ug, levels=[0.5], colors=['#00E676'], linewidths=0.8, linestyles='--')
        # Compact inline colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='4%', pad=0.03)
        cb  = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=6, length=2)
        cb.outline.set_linewidth(0.5)
        ax.set_title(f"AUROC = {auroc:.3f}", pad=3)
        _panel_border(ax)

        # ── Column 3: Open-world ───────────────────────────────────────
        ax = axes[row_i, 3]
        ax.imshow(vol, cmap='gray', interpolation='lanczos')
        # Render all known classes with the organ palette
        base_rgba = _seg_rgba(np.where(opn == 99, 0, opn))
        ax.imshow(base_rgba, interpolation='nearest')
        # Novel class 99 overlay
        novel_mask = (opn == 99)
        if novel_mask.any():
            novel_rgba = np.zeros((*novel_mask.shape, 4), dtype=np.float32)
            novel_rgba[novel_mask] = ORGAN_RGBA[99]
            ax.imshow(novel_rgba, interpolation='nearest')
        ax.contour(ug, levels=[0.5], colors=['#00E676'], linewidths=0.8, linestyles='--')
        _panel_border(ax)

        # Panel label (a), (b), (c) on the leftmost column
        axes[row_i, 0].text(
            0.02, 0.97, f"({panel_labels[row_i]})",
            transform=axes[row_i, 0].transAxes,
            fontsize=9, fontweight='bold', color='white',
            va='top', ha='left',
        )

    # Column headers (top row only)
    for col_i, title in enumerate(col_titles):
        axes[0, col_i].set_title(title, pad=5, fontsize=8, fontweight='bold')

    fig.patch.set_facecolor('white')
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    plt.rcParams.update(plt.rcParamsDefault)
    print(f"[NATURE-VIS] Figure saved to {output_path}")


def _panel_border(ax):
    """Apply minimal Nature-style panel formatting."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_edgecolor('#AAAAAA')


# ==========================================
# VISUALIZATION  (8 panels — debug)
# ==========================================
def visualize_full_pipeline(
    inputs, labels, pred_seg, raw_energy, gated_energy,
    open_pred, guidance_map, click_coord, slice_idx,
    adaptive_lambda, metrics, output_path
):
    fig, axes = plt.subplots(2, 4, figsize=(32, 14))
    axes = axes.flatten()

    # Slice all volumes at the same index (W axis for [D, H, W])
    vol_s    = np.rot90(inputs[:, :, slice_idx])
    lbl_s    = np.rot90(labels[:, :, slice_idx])
    pred_s   = np.rot90(pred_seg[:, :, slice_idx])
    energy_s = np.rot90(raw_energy[:, :, slice_idx])
    gated_s  = np.rot90(gated_energy[:, :, slice_idx])
    open_s   = np.rot90(np.asarray(open_pred)[:, :, slice_idx])
    unseen_gt = np.isin(lbl_s, UNSEEN_CLASSES)

    # ── Panel 0: CT Input ──
    axes[0].imshow(vol_s, cmap="gray")
    axes[0].set_title("CT Input")
    axes[0].axis("off")

    # ── Panel 1: Ground Truth ──
    axes[1].imshow(vol_s, cmap="gray")
    masked_lbl = np.ma.masked_where(lbl_s == 0, lbl_s)
    axes[1].imshow(masked_lbl, cmap="turbo", alpha=0.6, vmin=0, vmax=13)
    axes[1].contour(unseen_gt, colors='lime', linewidths=1.5, linestyles='dashed')
    axes[1].set_title("Ground Truth\n(Dashed lime = Unseen GT)")
    axes[1].axis("off")

    # ── Panel 2: Closed-Set Prediction ──
    axes[2].imshow(vol_s, cmap="gray")
    masked_pred = np.ma.masked_where(pred_s == 0, pred_s)
    axes[2].imshow(masked_pred, cmap="turbo", alpha=0.6)
    axes[2].contour(unseen_gt, colors='lime', linewidths=1.0, linestyles='dashed')
    axes[2].set_title("Closed-Set Pred\n(Misses Unseen)")
    axes[2].axis("off")

    # ── Panel 3: Raw Free Energy ──
    im3 = axes[3].imshow(energy_s, cmap="hot")
    axes[3].contour(unseen_gt, colors='lime', linewidths=1.5, linestyles='dashed')
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)
    axes[3].set_title(f"Raw Free Energy F(z)\n(λ = {adaptive_lambda:.3f})")
    axes[3].axis("off")

    # ── Panel 4: Gated Novelty Map  ReLU(F(z)-λ) * (1-p_bg) ──
    # vmin=0: force 0 = black so the colormap only shows genuine positive novelty.
    # Without this, matplotlib auto-scales to the float noise range (~±1e-3) and the
    # entire map renders as mid-orange even when all real detections are zero.
    gated_max = float(np.percentile(gated_s, 99.5)) if gated_s.max() > 0 else 0.1
    im4 = axes[4].imshow(gated_s, cmap="hot", vmin=0, vmax=max(gated_max, 1e-6))
    axes[4].contour(unseen_gt, colors='lime', linewidths=1.5, linestyles='dashed')
    plt.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)
    auroc_str  = f"{metrics.get('auroc', 0):.3f}" if metrics else "N/A"
    recall_str = f"{metrics.get('recall_at_lambda', 0):.3f}" if metrics else "N/A"
    axes[4].set_title(f"Gated Novelty Map\nAUROC={auroc_str}  Recall@λ={recall_str}")
    axes[4].axis("off")

    # ── Panel 5: User Guidance Map ──
    axes[5].imshow(vol_s, cmap="gray")
    if guidance_map is not None:
        g_s = np.rot90(guidance_map[:, :, slice_idx].cpu().numpy())
        axes[5].imshow(g_s, cmap="cool", alpha=0.6)
    # click_coord = (d, h, w); slice is [:, :, slice_idx] → plot axes are (h=col, d=row)
    if click_coord is not None and click_coord[2] == slice_idx:
        axes[5].plot(click_coord[1], click_coord[0],
                     marker='+', color='white', markersize=15, markeredgewidth=2)
    axes[5].set_title("User Guidance Map\n(Simulated Click)")
    axes[5].axis("off")

    # ── Panel 6: Open-World Result after Enrollment ──
    axes[6].imshow(vol_s, cmap="gray")
    masked_open = np.ma.masked_where(open_s == 0, open_s)
    axes[6].imshow(masked_open, cmap="turbo", alpha=0.6)
    axes[6].contour(open_s == 99, colors='red', linewidths=1.5)
    axes[6].contour(unseen_gt, colors='lime', linewidths=1.0, linestyles='dashed')
    axes[6].set_title("Open-World Result\n(Red outline = Enrolled Novel)")
    axes[6].axis("off")

    # ── Panel 7: Metrics summary ──
    axes[7].axis("off")
    if metrics:
        lines = [
            "OOD Detection Metrics",
            "─────────────────────",
            f"AUROC      {metrics.get('auroc', 0):.4f}",
            f"AUPRC      {metrics.get('auprc', 0):.4f}",
            f"Recall@λ   {metrics.get('recall_at_lambda', 0):.4f}",
            f"Precis@λ   {metrics.get('precision_at_lambda', 0):.4f}",
            f"F1@λ       {metrics.get('f1_at_lambda', 0):.4f}",
            "",
            f"TP {metrics.get('tp', 0):,}  FP {metrics.get('fp', 0):,}  FN {metrics.get('fn', 0):,}",
            f"N unseen   {metrics.get('n_unseen', 0):,}",
            f"N known    {metrics.get('n_known', 0):,}",
            f"λ          {adaptive_lambda:.4f}",
        ]
        axes[7].text(0.05, 0.95, "\n".join(lines),
                     transform=axes[7].transAxes, fontsize=11,
                     verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[7].set_title("Quantitative Results")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[VIS] Saved to {output_path}")


# ==========================================
# PREDICTOR CLOSURE
# ==========================================
def make_energy_predictor(model, memory_bank):
    """
    Wraps model + per-patch energy query into a single callable for
    sliding_window_inference.

    WHY THIS MATTERS:
    If you run sliding_window_inference(predictor=model) you get back
    averaged *embeddings*.  Averaging L2-normalised vectors shrinks their
    norm below 1, so every cosine similarity to a prototype drops uniformly
    and the free-energy range collapses (we saw -2.32 → -2.40, a range of
    only 0.08). Computing energy *inside* each patch and letting MONAI
    aggregate the scalar energy map instead avoids this entirely.

    Returns (logits [B,C,D,H,W], energy [B,1,D,H,W]) per patch.
    The extra channel dim on energy is required by MONAI's aggregation.
    """
    def predictor(patch):
        logits, embedding = model(patch)
        energy, _ = memory_bank.query_voxelwise_novelty(
            embedding.float(), include_background=False
        )
        return logits, energy.unsqueeze(1)   # [B,1,D,H,W] for MONAI cat/avg
    return predictor


# ==========================================
# MAIN INFERENCE LOOP
# ==========================================
def load_model(checkpoint_path, device, config):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = get_medopenseg(
        device=device,
        in_channels=config["model"]["in_channels"],
        out_channels=config["model"]["out_channels"],
        img_size=(96, 96, 96),
        feature_size=config["model"]["feature_size"],
        embed_dim_final=config["model"]["embed_dim_final"],
        pre_trained_weights=None,
    )
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint["model_state_dict"].items()}
    model.load_state_dict(state_dict)
    model.to(device).eval()
    return model


def infer_and_segment(model, memory_bank, test_loader, device, output_dir, n_samples=None):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    # adaptive_lambda was computed from training-time known-class energy distribution.
    # MUST use this rather than 0.0 — with energies around -2.3 to -2.5, a threshold
    # of 0.0 causes ReLU(energy - 0.0) = 0 everywhere (all energies are negative).
    adaptive_lambda = memory_bank.adaptive_lambda
    print(f"\n[INFO] adaptive_lambda = {adaptive_lambda:.4f}")
    if adaptive_lambda == 0.0:
        print("[WARNING] adaptive_lambda is 0.0 — memory bank may not have been saved after "
              "training, or the run was too short. Falling back to p95 of first batch energies.")

    all_results  = []
    all_metrics  = []
    all_vis_data = []   # for best-3 Nature figure

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            inputs = batch["image"].to(device)
            labels = batch["label"].to(device)

            filename = f"sample_{i}"
            if "image_meta_dict" in batch:
                filename = os.path.basename(
                    batch["image_meta_dict"]["filename_or_obj"][0]).split('.')[0]

            print(f"\n[PROC] ── {filename} ──")

            # ── Phase 1: Per-patch logits + energy ──
            # The predictor closure computes energy INSIDE each patch so MONAI
            # aggregates scalar energy values (not raw embeddings).  Averaging
            # embeddings before computing energy compresses the energy range to
            # ~0.08 nats and destroys OOD signal (seen as near-uniform hot map).
            predictor_fn = make_energy_predictor(model, memory_bank)
            with torch.amp.autocast(device.type):
                logits, raw_energy_5d = sliding_window_inference(
                    inputs, roi_size=(96, 96, 96), sw_batch_size=4, overlap=0.5,
                    mode="gaussian", predictor=predictor_fn,
                )

            # raw_energy_5d is [B, 1, D, H, W] (channel dim added inside predictor)
            raw_energy = raw_energy_5d.squeeze(1)   # [B, D, H, W]

            # ── Phase 1c: Per-volume lambda + gated novelty map S(z) (Eq. 8) ──
            # The training-time adaptive_lambda is calibrated on single 96³ patch energies.
            # At inference, MONAI aggregates overlapping patch energies by Gaussian averaging,
            # which reduces variance and shifts the distribution.  If volumes 2-5 have their
            # maximum energy below the trained lambda, ReLU fires nowhere → recall = 0.
            # Fix: recompute lambda as p95 of THIS volume's known-class voxel energies.
            p_bg     = torch.softmax(logits, dim=1)[:, 0]   # [B, D, H, W]
            seg_hard = torch.argmax(logits, dim=1)           # [B, D, H, W]

            vol_np    = inputs[0, 0].cpu().numpy()
            lbl_np    = labels[0, 0].cpu().numpy()
            pred_np   = seg_hard[0].cpu().numpy()
            energy_np = raw_energy[0].cpu().numpy()

            # Known-class mask: foreground (1-10), excluding unseen (11, 12, 13)
            known_np_mask = (lbl_np > 0) & (~np.isin(lbl_np, UNSEEN_CLASSES))
            if known_np_mask.any():
                per_vol_lambda = float(np.percentile(energy_np[known_np_mask], 95))
            else:
                per_vol_lambda = adaptive_lambda

            print(f"[INFO] lambda:  trained={adaptive_lambda:.4f}  "
                  f"per-vol={per_vol_lambda:.4f}  "
                  f"delta={per_vol_lambda - adaptive_lambda:+.4f}")

            # S(z) = (1 − p_bg) · ReLU(F(z) − λ_vol)
            gated_energy = (1.0 - p_bg) * torch.relu(raw_energy - per_vol_lambda)
            gated_np     = gated_energy[0].cpu().numpy()

            # ── Quantitative OOD metrics (before any enrollment) ──
            metrics = compute_novelty_metrics(energy_np, lbl_np, per_vol_lambda)

            # ── Phase 2: Interactive enrollment ──
            # After enrolling (mu_new, kappa_new), re-query the memory bank with the full
            # volume embedding and take the nearest-prototype assignment.  This correctly
            # implements "pulling all semantically similar voxels into the new well" (paper
            # Section 3.5) rather than thresholding the energy map (which has float16 noise
            # everywhere, painting the entire body red).
            # Gate with (1 - p_bg) > 0.5 to suppress air / true background.
            open_world_np    = pred_np.copy()
            guidance_map     = None
            already_enrolled = (99 in memory_bank.prototypes)
            click_coord      = simulate_user_click(lbl_np)

            need_embedding = click_coord or already_enrolled
            embedding = None

            if need_embedding:
                with torch.no_grad(), torch.amp.autocast(device.type):
                    _, embedding = sliding_window_inference(
                        inputs, roi_size=(96, 96, 96), sw_batch_size=4, overlap=0.5,
                        mode="gaussian", predictor=model,
                    )

            if click_coord and not already_enrolled:
                print(f"[ENROLL] Novel anomaly detected. Simulating user click at {click_coord}")
                # Gradient-modulated Gaussian guidance (paper Eq. 9)
                guidance_map = create_guidance_map(
                    lbl_np.shape, click_coord, vol_np=vol_np, sigma=6.0, beta=5.0
                ).to(device)
                mu_new, kappa_new = enroll_new_class(embedding, guidance_map, device)
                memory_bank.enroll_interactive_prototype(
                    new_class_id=99, mu_new=mu_new, kappa_new=kappa_new,
                )
                print(f"[ENROLL] Enrolled class 99: kappa={kappa_new.item():.1f}")
            elif already_enrolled:
                print("[ENROLL] Class 99 already enrolled from previous patient — re-querying.")

            # Re-query memory bank (now containing class 99) using the averaged embedding.
            # mu_new was computed from the same averaged embedding, so cosine similarities
            # are internally consistent.  Gate by body mask to suppress air voxels.
            if embedding is not None and (99 in memory_bank.prototypes):
                _, new_pred = memory_bank.query_voxelwise_novelty(
                    embedding.float(), include_background=False
                )
                new_pred_np  = new_pred[0].cpu().numpy()
                body_fg_mask = (p_bg[0].cpu().numpy() < 0.5)   # foreground only
                novel_voxels = (new_pred_np == 99) & body_fg_mask
                open_world_np[novel_voxels] = 99
                print(f"[ENROLL] Class 99 assigned to {novel_voxels.sum():,} voxels "
                      f"({100*novel_voxels.mean():.2f}% of volume).")
            elif not click_coord:
                print("[ENROLL] No unseen voxels found in this volume.")

            # ── Visualization ──
            slice_idx = find_slice_with_most_unseen(lbl_np)

            # Debug figure (8 panels with diagnostics)
            vis_path = os.path.join(output_dir, f"{filename}_pipeline.png")
            visualize_full_pipeline(
                inputs=vol_np, labels=lbl_np, pred_seg=pred_np,
                raw_energy=energy_np, gated_energy=gated_np,
                open_pred=open_world_np,
                guidance_map=guidance_map, click_coord=click_coord,
                slice_idx=slice_idx, adaptive_lambda=per_vol_lambda,
                metrics=metrics, output_path=vis_path,
            )

            # Per-volume 4-panel publication figure
            pub_path = os.path.join(output_dir, f"{filename}_publication.png")
            visualize_publication(
                inputs=vol_np, labels=lbl_np, pred_seg=pred_np,
                gated_energy=gated_np, open_pred=open_world_np,
                slice_idx=slice_idx, metrics=metrics, output_path=pub_path,
            )

            all_results.append(open_world_np)
            if metrics:
                all_metrics.append(metrics)

            # Accumulate data for the best-3 Nature figure
            all_vis_data.append(dict(
                vol_np=vol_np, lbl_np=lbl_np, pred_np=pred_np,
                gated_np=gated_np, open_np=open_world_np,
                slice_idx=slice_idx, metrics=metrics, filename=filename,
            ))

            if n_samples is not None and (i + 1) >= n_samples:
                print(f"[INFO] --debug: stopping after {n_samples} samples.")
                break

    # ── Aggregate metrics + LaTeX table (per-class mean±std) ──
    if all_metrics:
        print("\n[SUMMARY] === Aggregate OOD Metrics ===")
        for key in ["auroc", "auprc", "recall_at_lambda", "f1_at_lambda"]:
            vals = [m[key] for m in all_metrics if key in m]
            if vals:
                print(f"  {key:25s}: mean={np.mean(vals):.4f}  std={np.std(vals):.4f}")
        print("[SUMMARY] =====================================\n")

        latex_path = os.path.join(output_dir, "ood_results_table.tex")
        generate_latex_table(all_metrics, latex_path)

    # ── Nature-style best-3 qualitative figure ──
    if all_vis_data:
        nature_path = os.path.join(output_dir, "best3_nature_figure.png")
        visualize_best_examples(all_vis_data, nature_path, n_best=3)

    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config_btcv")
    parser.add_argument("--exp",    type=str, default="btcv/btcv_fix2")
    parser.add_argument("--debug",  action="store_true",
                        help="Run on first 5 validation volumes only (fast iteration).")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Override number of test samples (default: all in validation set).")
    parser.add_argument("--use_last", action="store_true",
                        help="Use checkpoint_last.pth + energy_memory_bank.pth (same step, "
                             "no model/bank mismatch). Recommended when best_energy_memory_bank.pth "
                             "is stale (e.g. kappas changed since last best-Dice checkpoint).")
    args = parser.parse_args()

    n_samples = 5 if args.debug else args.n_samples
    if args.debug:
        print("[INFO] --debug mode: using first 5 validation samples.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config_path = os.path.join("./configs", f"{args.config}.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    exps_root       = "/home/vargas/openwell/outputs"
    best_mb_path = os.path.join(exps_root, args.exp, "best_energy_memory_bank.pth")
    last_mb_path = os.path.join(exps_root, args.exp, "energy_memory_bank.pth")

    if args.use_last:
        checkpoint_path = os.path.join(exps_root, args.exp, "checkpoint_last.pth")
        mb_path = last_mb_path
        print("[INFO] --use_last: loading checkpoint_last.pth + energy_memory_bank.pth "
              "(guaranteed same-step, no model/bank kappa mismatch)")
    else:
        checkpoint_path = os.path.join(exps_root, args.exp, "best_checkpoint.pth")
        mb_path = best_mb_path if os.path.exists(best_mb_path) else last_mb_path
        if not os.path.exists(best_mb_path):
            print("[WARNING] best_energy_memory_bank.pth not found — loading energy_memory_bank.pth "
                  "which may be from a different training step (kappa mismatch). "
                  "Use --use_last for a guaranteed matched pair.")

    model = load_model(checkpoint_path, device, config)

    embed_dim   = config["model"]["embed_dim_final"]
    memory_bank = MemoryBankV(memory_size=100, feature_dim=embed_dim).to(device)
    memory_bank.load_memory_bank(mb_path, device=device)

    data_dir  = config["data"]["data_dir"]
    datasets  = os.path.join(data_dir, config["data"]["split_json"])
    _, _, test_transforms = load_transforms(config, device)

    all_val_files = load_decathlon_datalist(datasets, True, "validation")
    test_files    = all_val_files[:n_samples] if n_samples else all_val_files
    print(f"[INFO] Running inference on {len(test_files)} validation volumes "
          f"({'debug' if args.debug else 'full'} mode).")
    test_ds     = Dataset(data=test_files, transform=test_transforms)
    test_loader = ThreadDataLoader(test_ds, batch_size=1, num_workers=0)

    output_vis_dir = os.path.join(exps_root, args.exp, "vis_interactive")
    results = infer_and_segment(model, memory_bank, test_loader, device, output_vis_dir,
                                n_samples=n_samples)

    if not results:
        return

    output_seg_dir = "output_segmentations"
    os.makedirs(output_seg_dir, exist_ok=True)
    for i, seg in enumerate(results):
        nib.save(
            nib.Nifti1Image(np.asarray(seg).astype(np.int16), affine=np.eye(4)),
            os.path.join(output_seg_dir, f"segmentation_{i}.nii.gz"),
        )


if __name__ == "__main__":
    main()
