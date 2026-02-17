import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import GeneralizedDiceLoss, DiceLoss

class OpenSetDiceCELoss(nn.Module):
    """
    Hybrid Loss: Cross Entropy + Generalized Dice (Protects small organs).
    """
    def __init__(self, ignore_index=255, lambda_dice=1.0, lambda_ce=1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce
        
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        
        # CHANGED: GeneralizedDiceLoss helps small organs (IVC, Aorta) survive the 'Shock'
        self.dice = GeneralizedDiceLoss(
            include_background=False, 
            to_onehot_y=True,
            softmax=True
        )

    def forward(self, preds, target):
        # CE Target
        if target.dtype != torch.long:
            target = target.long()
        target_ce = target.squeeze(1) if target.ndim == preds.ndim else target
        ce_loss = self.ce(preds, target_ce)
        
        # Dice Target (Map 255 -> 0, then ignore class 0)
        target_dice = target.clone()
        target_dice[target == self.ignore_index] = 0
        if target_dice.ndim == preds.ndim - 1:
             target_dice = target_dice.unsqueeze(1)

        dice_loss = self.dice(preds, target_dice)
        
        return self.lambda_ce * ce_loss + self.lambda_dice * dice_loss

def compute_vmf_loss(embeddings, labels, memory_bank, temperature=0.1, ignore_index=255):
    """
    Computes vMF Energy Loss with Unseen Rejection.
    """
    if len(memory_bank.prototypes) == 0:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    # 1. Prepare Prototypes
    classes = sorted(memory_bank.prototypes.keys())
    mus = torch.stack([memory_bank.prototypes[c]['mu'] for c in classes]).to(embeddings.device)
    kappas = torch.stack([memory_bank.prototypes[c]['kappa'] for c in classes]).to(embeddings.device).unsqueeze(1)
    
    # 2. Vectorized Lookup
    if not classes:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
    max_cls = int(max(classes))
    lookup = torch.full((max_cls + 1,), -1, device=embeddings.device, dtype=torch.long)
    for idx, c in enumerate(classes):
        lookup[int(c)] = idx

    B, F_dim = embeddings.shape[:2]
    emb_flat = embeddings.reshape(B, F_dim, -1).permute(0, 2, 1)
    lbl_flat = labels.reshape(B, -1)
    emb_flat = F.normalize(emb_flat, p=2, dim=2)

    total_loss = 0.0
    valid_batches = 0

    for b in range(B):
        batch_lbls = lbl_flat[b]
        
        # --- PART A: Attraction (Known Classes) ---
        valid_mask = (batch_lbls != ignore_index) & (batch_lbls <= max_cls)
        
        if valid_mask.any():
            valid_lbls = batch_lbls[valid_mask]
            target_indices = lookup[valid_lbls.long()]
            final_mask = target_indices != -1
            
            if final_mask.any():
                final_targets = target_indices[final_mask]
                final_embs = emb_flat[b][valid_mask][final_mask]

                cos_sim = torch.matmul(final_embs, mus.t())
                logits = cos_sim * kappas.view(1, -1)
                
                total_loss += F.cross_entropy(logits / temperature, final_targets)

        # --- PART B: Rejection (The Open-World Fix) ---
        # Explicitly push '255' pixels AWAY from all prototypes
        unseen_mask = (batch_lbls == ignore_index)
        
        if unseen_mask.any():
            # Subsample if too many pixels to save memory
            unseen_embs = emb_flat[b][unseen_mask]
            if unseen_embs.shape[0] > 10000:
                 perm = torch.randperm(unseen_embs.size(0))[:10000]
                 unseen_embs = unseen_embs[perm]

            # Cosine Sim to ALL prototypes
            unseen_sim = torch.matmul(unseen_embs, mus.t()) 
            
            # Penalty: If similarity > 0.1, punish it.
            # This creates the "Energy Well" separation.
            #rejection_penalty = F.relu(unseen_sim - 0.1).mean()
            rejection_penalty = F.relu(unseen_sim - 0.05).mean()
            total_loss += 5.0 * rejection_penalty
            
            # Weight 0.5 is usually enough
            #total_loss += 0.5 * rejection_penalty

        valid_batches += 1

    if valid_batches > 0:
        return total_loss / valid_batches
    else:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)