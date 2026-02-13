import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss

class OpenSetDiceCELoss(nn.Module):
    """
    Hybrid Loss that strictly ignores 'Unseen' pixels (mapped to 255).
    """
    def __init__(self, ignore_index=255, lambda_dice=1.0, lambda_ce=1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.lambda_dice = lambda_dice
        self.lambda_ce = lambda_ce
        
        # 1. CE: Natively supports ignore_index
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        
        # 2. Dice: We exclude background calculation so the pixels 
        #    mapped to 0 don't contribute to the background class gradient.
        self.dice = DiceLoss(
            include_background=False, 
            to_onehot_y=True,
            softmax=True
        )

    def forward(self, preds, target):
        # CE Target Prep
        if target.dtype != torch.long:
            target = target.long()
        target_ce = target.squeeze(1) if target.ndim == preds.ndim else target

        ce_loss = self.ce(preds, target_ce)
        
        # Dice Target Prep (Map 255 -> 0, then ignore class 0)
        target_dice = target.clone()
        target_dice[target == self.ignore_index] = 0
        if target_dice.ndim == preds.ndim - 1:
             target_dice = target_dice.unsqueeze(1)

        dice_loss = self.dice(preds, target_dice)
        
        return self.lambda_ce * ce_loss + self.lambda_dice * dice_loss

def compute_vmf_loss(embeddings, labels, memory_bank, temperature=0.1, ignore_index=255):
    """
    Computes vMF Energy Loss (Vectorized & Optimized).
    """
    if len(memory_bank.prototypes) == 0:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    # 1. Prepare Prototypes
    classes = sorted(memory_bank.prototypes.keys())
    mus = torch.stack([memory_bank.prototypes[c]['mu'] for c in classes]).to(embeddings.device)
    kappas = torch.stack([memory_bank.prototypes[c]['kappa'] for c in classes]).to(embeddings.device).unsqueeze(1)
    
    # 2. Vectorized Lookup Table (Fixes speed & TypeError)
    if not classes:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        
    # FIX: Explicitly cast to int for torch.full size
    max_cls = int(max(classes))
    
    # Create a lookup tensor initialized to -1
    lookup = torch.full((max_cls + 1,), -1, device=embeddings.device, dtype=torch.long)
    
    # Populate lookup: Class ID -> Prototype Index
    for idx, c in enumerate(classes):
        lookup[int(c)] = idx

    B, F_dim = embeddings.shape[:2]
    # Flatten: [B, F, N] -> [B, N, F]
    emb_flat = embeddings.reshape(B, F_dim, -1).permute(0, 2, 1)
    lbl_flat = labels.reshape(B, -1)
    
    # Normalize image embeddings once
    emb_flat = F.normalize(emb_flat, p=2, dim=2)

    loss = 0.0
    valid_batches = 0

    for b in range(B):
        batch_lbls = lbl_flat[b]
        
        # Fast Masking
        # A. Ignore 'ignore_index' (255)
        # B. Ignore labels larger than our lookup table (safety check)
        valid_mask = (batch_lbls != ignore_index) & (batch_lbls <= max_cls)
        
        if not valid_mask.any(): 
            continue

        # Get valid pixels
        valid_lbls = batch_lbls[valid_mask]
        
        # Map Labels to Prototype Indices using GPU Lookup
        # FIX: Ensure valid_lbls is LongTensor for indexing
        target_indices = lookup[valid_lbls.long()]
        
        # Filter out classes that are not yet in the memory bank (mapped to -1)
        final_mask = target_indices != -1
        
        if not final_mask.any():
            continue
            
        final_targets = target_indices[final_mask]
        # Select corresponding embeddings
        final_embs = emb_flat[b][valid_mask][final_mask]

        # 3. Compute Logits & Loss
        # Cosine Sim: [N_valid, F] @ [F, N_classes] -> [N_valid, N_classes]
        cos_sim = torch.matmul(final_embs, mus.t())
        logits = cos_sim * kappas.view(1, -1)
        
        loss += F.cross_entropy(logits / temperature, final_targets)
        valid_batches += 1

    if valid_batches > 0:
        return loss / valid_batches
    else:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)