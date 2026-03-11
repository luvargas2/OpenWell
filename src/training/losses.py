import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import GeneralizedDiceLoss, DiceLoss

class OpenSetDiceCELoss(nn.Module):
    """
    Hybrid Loss: Cross Entropy + Generalized Dice.
    Now accepts 'ce_weight' to handle class imbalance (e.g. boosting small organs).
    """
    def __init__(self, ignore_index=255, lambda_dice=1.0, lambda_ce=1.0, ce_weight=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce
        
        # Initialize CE with optional class weights
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, weight=ce_weight)
        
        # GeneralizedDiceLoss is critical for small organs vs large Liver
        self.dice = GeneralizedDiceLoss(
            include_background=False, 
            to_onehot_y=True,
            softmax=True
        )

    def forward(self, preds, target):
        # CE Target (Long Tensor)
        if target.dtype != torch.long:
            target = target.long()
        
        # Handle [B, 1, D, H, W] vs [B, D, H, W] mismatch
        target_ce = target.squeeze(1) if target.ndim == preds.ndim else target
        
        ce_loss = self.ce(preds, target_ce)
        
        # Dice Target (Map 255 -> 0, then ignore class 0 in metric)
        target_dice = target.clone()
        target_dice[target == self.ignore_index] = 0
        
        # Ensure channel dim exists for Dice
        if target_dice.ndim == preds.ndim - 1:
             target_dice = target_dice.unsqueeze(1)

        dice_loss = self.dice(preds, target_dice)
        
        return self.lambda_ce * ce_loss + self.lambda_dice * dice_loss


def compute_vmf_loss(
    embeddings,            # [B, F, D, H, W]
    labels,                # [B, 1, D, H, W] or [B, D, H, W]
    memory_bank,
    tau=1.0,               
    ignore_index=255,
    reject_weight=1.0,     # Weight for pushing away unknown pixels
    max_unseen=10000,
    include_background=False
):
    """
    vMF Energy Loss.
    
    1. Attraction (Known): Pulls known pixels towards their prototype (mu).
    2. Rejection (Unknown): Pushes 'ignore_index' pixels away from ALL prototypes.
       Minimizing logsumexp(logits) is equivalent to Maximizing Free Energy.
    """

    # --- Safety Checks ---
    if len(memory_bank.prototypes) == 0:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    # 1. Select Classes (Usually exclude background for vMF attraction)
    classes = sorted(memory_bank.prototypes.keys())
    if not include_background:
        classes = [c for c in classes if int(c) != 0]

    if len(classes) == 0:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    # 2. Prepare Prototypes (Mus & Kappas)
    mus = torch.stack([memory_bank.prototypes[c]["mu"] for c in classes]).to(embeddings.device)  # [C, F]
    kappas = torch.stack([memory_bank.prototypes[c]["kappa"] for c in classes]).to(embeddings.device)  # [C]
    kappas = torch.clamp(kappas, min=1.0, max=30.0)
    mus = F.normalize(mus, p=2, dim=1)

    # 3. Create Label -> Index Lookup Table
    max_cls = int(max(int(c) for c in classes))
    lookup = torch.full((max_cls + 1,), -1, device=embeddings.device, dtype=torch.long)
    for idx, c in enumerate(classes):
        ci = int(c)
        if ci <= max_cls:
            lookup[ci] = idx

    # 4. Flatten Embeddings & Labels
    if labels.dim() == embeddings.dim():
        labels_ = labels.squeeze(1)
    else:
        labels_ = labels

    B, F_dim = embeddings.shape[:2]
    emb_flat = embeddings.reshape(B, F_dim, -1).permute(0, 2, 1)   # [B, N, F]
    lbl_flat = labels_.reshape(B, -1)                               # [B, N]
    emb_flat = F.normalize(emb_flat, p=2, dim=2)

    total_loss = 0.0
    valid_batches = 0

    for b in range(B):
        batch_lbls = lbl_flat[b]     # [N]
        batch_embs = emb_flat[b]     # [N, F]

        # -------------------
        # PART A: Attraction (Known Classes)
        # -------------------
        valid_mask = (batch_lbls != ignore_index) & (batch_lbls >= 0) & (batch_lbls <= max_cls)
        attr_loss = 0.0
        has_attr = False

        if valid_mask.any():
            valid_lbls = batch_lbls[valid_mask].long()
            target_idx = lookup[valid_lbls]          
            keep = target_idx != -1

            if keep.any():
                z = batch_embs[valid_mask][keep]     # [Na, F]
                y = target_idx[keep]                 # [Na]

                cos = z @ mus.t()                    # [Na, C]
                logits = (cos * kappas.view(1, -1)) / tau
                attr_loss = F.cross_entropy(logits, y)
                has_attr = True

        # -------------------
        # PART B: Rejection (Unknown Classes)
        # -------------------
        rej_loss = 0.0
        has_rej = False

        unseen_mask = (batch_lbls == ignore_index)
        if unseen_mask.any():
            zu = batch_embs[unseen_mask]             # [Nu, F]

            # Subsample to save memory/compute
            if zu.shape[0] > max_unseen:
                perm = torch.randperm(zu.size(0), device=zu.device)[:max_unseen]
                zu = zu[perm]

            cos_u = zu @ mus.t()                      # [Nu, C]
            logits_u = (cos_u * kappas.view(1, -1)) / tau

            # Minimizing LogSumExp => Pushing away from all clusters
            rej_loss = torch.logsumexp(logits_u, dim=1).mean()
            has_rej = True

        # Combine
        if has_attr or has_rej:
            # Note: reject_weight scales how strongly we push "unseen" away
            loss_b = (attr_loss if has_attr else 0.0) + (reject_weight * rej_loss if has_rej else 0.0)
            total_loss += loss_b
            valid_batches += 1

    if valid_batches > 0:
        return total_loss / valid_batches
    return torch.tensor(0.0, device=embeddings.device, requires_grad=True)