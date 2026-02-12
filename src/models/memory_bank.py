import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from adjustText import adjust_text 
import numpy as np

class MemoryBankV(nn.Module):
    """
    Energy-Based Memory Bank for Open-World 3D Segmentation.
    
    Models each class as a Von Mises-Fisher (vMF) distribution with:
    1) Mean Direction (mu): The prototype vector.
    2) Concentration (kappa): The inverse variance (how 'tight' the class is).
    
    Novelty is computed via Free Energy: pixels that don't fall into 
    any class's 'energy well' have high energy -> Unknown.
    """

    def __init__(
        self,
        feature_dim: int,
        memory_size: int = 100,
        epoch_counter = 0,
        alpha: float = 0.9, # Momentum for EMA updates
        save_path: str = "./prototypes"
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.memory_size = memory_size
        self.alpha = alpha
        self.epoch_counter = epoch_counter
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

        # Storage: {class_id: {'mu': Tensor(F), 'kappa': Tensor(1), 'count': int}}
        self.prototypes = {}
        self.UNKNOWN_CLASS_ID = 999

    def save_memory_bank(self, save_path):
        if not self.prototypes:
            return
        # Save dict with tensors on CPU
        save_dict = {}
        for k, v in self.prototypes.items():
            save_dict[k] = {
                'mu': v['mu'].detach().cpu(),
                'kappa': v['kappa'].detach().cpu(),
                'count': v['count']
            }
        torch.save(save_dict, save_path)
        print(f"[INFO] Prototypes saved to {save_path}")

    def load_memory_bank(self, memory_bank_path, device="cuda"):
        if not os.path.exists(memory_bank_path):
            print(f"[ERROR] File not found: {memory_bank_path}")
            return
        checkpoint = torch.load(memory_bank_path, map_location="cpu",weights_only=False)
        self.prototypes = {}
        for k, v in checkpoint.items():
            self.prototypes[k] = {
                'mu': v['mu'].to(device),
                'kappa': v['kappa'].to(device),
                'count': v['count']
            }
        print(f"[INFO] Loaded {len(self.prototypes)} classes (with Energy parameters).")

    def update_prototypes(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        Updates Mean (mu) and Concentration (kappa).
        CRITICAL FIX: We clamp Background (Class 0) to be 'loose' (Low Kappa).
        """
        B, F_dim, D, H, W = embeddings.shape
        embeddings_flat = embeddings.view(B, F_dim, -1).permute(0, 2, 1) # [B, N, F]
        labels_flat = labels.view(B, -1) # [B, N]

        for b_idx in range(B):
            lbls = labels_flat[b_idx]
            feats = embeddings_flat[b_idx] # [N, F]
            
            unique_cls = lbls.unique()
            for cls_id in unique_cls:
                cval = cls_id.item()
                
                # --- LOGIC FIX: Handle Ignore Label if present ---
                if cval == self.UNKNOWN_CLASS_ID or cval == 255: 
                    continue

                mask = (lbls == cval)
                class_feats = feats[mask]
                
                if class_feats.shape[0] < 5: continue 

                # 1. Normalize
                class_feats = F.normalize(class_feats, p=2, dim=1)

                # 2. Mean Direction
                mean_vector = class_feats.mean(dim=0)
                mean_dir = F.normalize(mean_vector, p=2, dim=0)

                # 3. Estimate Kappa
                R = mean_vector.norm(p=2).clamp(min=1e-6, max=0.999)
                batch_kappa = (R * self.feature_dim) / (1 - R**2)
                
                # --- CRITICAL FIX: The "Soft Background" Logic ---
                # Background (0) must have a wide well (low kappa) to avoid overfitting to unseen objects.
                # Organs (1+) can have tight wells.
                if cval == 0:
                    batch_kappa = torch.clamp(batch_kappa, max=10.0) # FORCE SHALLOW WELL
                else:
                    batch_kappa = torch.clamp(batch_kappa, max=100.0) # Allow deep wells for organs

                # 4. Update Memory (EMA)
                if cval not in self.prototypes:
                    self.prototypes[cval] = {
                        'mu': mean_dir.detach(),
                        'kappa': batch_kappa.detach(),
                        'count': 1
                    }
                else:
                    old_mu = self.prototypes[cval]['mu']
                    old_kappa = self.prototypes[cval]['kappa']
                    
                    new_mu = self.alpha * old_mu + (1 - self.alpha) * mean_dir
                    new_mu = F.normalize(new_mu, p=2, dim=0)
                    new_kappa = self.alpha * old_kappa + (1 - self.alpha) * batch_kappa
                    
                    self.prototypes[cval]['mu'] = new_mu
                    self.prototypes[cval]['kappa'] = new_kappa
                    self.prototypes[cval]['count'] += 1

        if self.epoch_counter % 10 == 0:
            self.save_tsne_plot()

            
    def query_voxelwise_novelty(self, embedding_3d: torch.Tensor):
        """
        Returns:
            energy_map: [B, D, H, W] - Higher value means MORE NOVEL (Unknown).
            class_pred: [B, D, H, W] - Predicted class ID (based on best energy well).
        """
        if not self.prototypes:
            # If empty, everything is unknown (high energy)
            return torch.ones_like(embedding_3d[:,0]), torch.zeros_like(embedding_3d[:,0])

        B, F_dim, D, H, W = embedding_3d.shape
        
        # 1. Prepare Prototypes
        # Stack mu: [C, F]
        # Stack kappa: [C, 1]
        classes = sorted(self.prototypes.keys())
        mus = torch.stack([self.prototypes[c]['mu'] for c in classes]).to(embedding_3d.device)
        kappas = torch.stack([self.prototypes[c]['kappa'] for c in classes]).to(embedding_3d.device).unsqueeze(1)
        
        # 2. Flatten Image Embeddings: [B, N, F]
        emb_flat = embedding_3d.view(B, F_dim, -1).permute(0, 2, 1)
        emb_flat = F.normalize(emb_flat, p=2, dim=2) # Normalize to sphere

        # 3. Compute vMF Logits (Similarity scaled by Concentration)
        # Cosine Similarity: [B, N, F] @ [F, C] -> [B, N, C]
        cosine_sim = torch.matmul(emb_flat, mus.t())
        
        # Scale by Kappa (The "Energy Well" depth)
        # Tighter classes (high kappa) punish deviation more
        logits = cosine_sim * kappas.view(1, 1, -1) # [B, N, C]
        
        # 4. Compute Free Energy
        # E(z) = - LogSumExp(logits)
        # We negate it so High Energy = Low Probability (Novel)
        energy_flat = -torch.logsumexp(logits, dim=2) # [B, N]
        
        # 5. Get Predictions (Max Logit = Best Well)
        val, idx = torch.max(logits, dim=2)
        pred_flat = torch.tensor(classes, device=embedding_3d.device)[idx] # Map indices back to class IDs
        
        # 6. Reshape
        energy_map = energy_flat.view(B, D, H, W)
        class_pred = pred_flat.view(B, D, H, W)

        return energy_map, class_pred

    def save_tsne_plot(self, perplexity=30.0, random_state=42):
        if len(self.prototypes) < 2: return

        # Extract Mus
        class_ids = sorted(self.prototypes.keys())
        embeddings = torch.stack([self.prototypes[c]['mu'] for c in class_ids]).cpu().numpy()
        
        # Extract Kappas for point size (Tight classes = Larger points)
        kappas = torch.stack([self.prototypes[c]['kappa'] for c in class_ids]).cpu().numpy()
        # Normalize kappas for display size
        sizes = 100 + (kappas - kappas.min()) / (kappas.max() - kappas.min() + 1e-6) * 200

        tsne = TSNE(n_components=2, perplexity=min(30, len(class_ids)-1), random_state=random_state)
        embeddings_2d = tsne.fit_transform(embeddings)

        sns.set_style("whitegrid")
        plt.figure(figsize=(8, 6))
        colors = sns.color_palette("husl", n_colors=len(class_ids))

        texts = []
        for i, (x, y) in enumerate(embeddings_2d):
            plt.scatter(x, y, color=colors[i], s=sizes[i], edgecolors='k', alpha=0.8, label=str(class_ids[i]))
            texts.append(plt.text(x, y, str(class_ids[i]), fontsize=10, fontweight='bold'))

        adjust_text(texts)
        plt.title("Prototype Energy Wells (Size ~ Concentration)", fontsize=14)
        
        plot_filename = os.path.join(self.save_path, f"prototypes_energy_{self.epoch_counter}.png")
        plt.savefig(plot_filename, dpi=300)
        plt.close()