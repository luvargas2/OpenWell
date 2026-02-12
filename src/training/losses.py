import torch
import torch.nn as nn
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
        
        # 2. Dice: We exclude background calculation so the mapped pixels 
        #    (mapped 255->0) don't contribute to the gradient.
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
    Computes vMF Energy Loss.
    Skips pixels labeled as ignore_index (Unseen).
    """
    if len(memory_bank.prototypes) == 0:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    classes = sorted(memory_bank.prototypes.keys())
    mus = torch.stack([memory_bank.prototypes[c]['mu'] for c in classes]).to(embeddings.device)
    kappas = torch.stack([memory_bank.prototypes[c]['kappa'] for c in classes]).to(embeddings.device).unsqueeze(1)
    class_indices = {c: i for i, c in enumerate(classes)}

    B, F_dim = embeddings.shape[:2]
    # Flatten: [B, F, N] -> [B, N, F]
    emb_flat = embeddings.reshape(B, F_dim, -1).permute(0, 2, 1)
    lbl_flat = labels.reshape(B, -1)
    
    emb_flat = torch.nn.functional.normalize(emb_flat, p=2, dim=2)

    loss = 0.0
    valid_batches = 0

    for b in range(B):
        # Filter: Must be a known class AND not the ignore index
        # This prevents the model from pulling 'Unseen' pixels towards ANY prototype.
        valid_mask = torch.isin(lbl_flat[b], torch.tensor(classes, device=embeddings.device)) & (lbl_flat[b] != ignore_index)
        
        if not valid_mask.any(): continue

        valid_emb = emb_flat[b][valid_mask]
        valid_lbl = lbl_flat[b][valid_mask]

        target_indices = torch.tensor([class_indices[l.item()] for l in valid_lbl], device=embeddings.device)

        # Logits = Similarity * Concentration
        cos_sim = torch.matmul(valid_emb, mus.t())
        logits = cos_sim * kappas.view(1, -1)
        
        loss += torch.nn.functional.cross_entropy(logits / temperature, target_indices)
        valid_batches += 1

    if valid_batches > 0:
        return loss / valid_batches
    else:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)