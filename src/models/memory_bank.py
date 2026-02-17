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
    
    Theory:
    Models each class as a Von Mises-Fisher (vMF) distribution on the hypersphere.
    - Mean (mu): The semantic center of the class.
    - Concentration (kappa): Inverse variance. High kappa = Tight cluster (Deep Energy Well).
    
    Open-Set Logic:
    - We clamp the Background (Class 0) kappa to be low (Shallow Well).
    - We ignore 'Unseen' pixels (Label 255) during updates so they don't pollute prototypes.
    """

    def __init__(
        self,
        feature_dim: int,
        memory_size: int = 100,
        epoch_counter: int = 0,
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
        self.IGNORE_LABEL = 255  # Consistent with transforms.py

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
            print(f"[WARNING] Memory bank file not found: {memory_bank_path}. Starting fresh.")
            return
        
        # weights_only=False allows loading complex dict structures
        checkpoint = torch.load(memory_bank_path, map_location=device, weights_only=False)
        self.prototypes = {}
        for k, v in checkpoint.items():
            self.prototypes[k] = {
                'mu': v['mu'].to(device),
                'kappa': v['kappa'].to(device),
                'count': v['count']
            }
        print(f"[INFO] Loaded {len(self.prototypes)} classes into Memory Bank.")

    @torch.no_grad()
    def update_prototypes(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        Updates Mean (mu) and Concentration (kappa) using online statistics (EMA).
        """
        # Embeddings: [B, F, D, H, W] -> Flatten to [B, N_voxels, F]
        B, F_dim, D, H, W = embeddings.shape
        embeddings_flat = embeddings.view(B, F_dim, -1).permute(0, 2, 1) 
        labels_flat = labels.view(B, -1)

        for b_idx in range(B):
            lbls = labels_flat[b_idx]
            feats = embeddings_flat[b_idx]
            
            unique_cls = lbls.unique()
            for cls_id in unique_cls:
                cval = cls_id.item()
                
                # --- LOGIC FIX 1: Ignore Unseen/Void Pixels ---
                # We strictly skip updates for the 'Unseen' mapped label (255)
                # and the inference-time 'Unknown' label (999).
                if cval == self.UNKNOWN_CLASS_ID or cval == self.IGNORE_LABEL: 
                    continue

                mask = (lbls == cval)
                class_feats = feats[mask]
                
                # Minimum voxels needed for statistical stability
                if class_feats.shape[0] < 5: 
                    continue 

                # 1. Normalize features (project to hypersphere)
                class_feats = F.normalize(class_feats, p=2, dim=1)

                # 2. Calculate Mean Direction
                mean_vector = class_feats.mean(dim=0)
                mean_dir = F.normalize(mean_vector, p=2, dim=0)

                # 3. Estimate Concentration (Kappa)
                # R = length of mean vector. R close to 1 = Tight cluster.
                R = mean_vector.norm(p=2).clamp(min=1e-6, max=0.999)
                
                # vMF approximation for Kappa
                batch_kappa = (R * self.feature_dim) / (1 - R**2 + 1e-6)
                
                # --- LOGIC FIX 2: Soft Background ---
                # Background (0) is a "garbage" class with high variance. 
                # We clamp its kappa to be low (max 10) to create a 'Shallow Well'.
                # This allows Unseen objects (which don't match organs) to 
                # register as High Energy rather than falling into the Background well.

                if cval == 0:
                    # Deepen the Background well so it captures the "true" background makes the background "well" deeper and narrower.
                    # effectively leaving the "Unseen" pixels stranded in high-energy space.
                    batch_kappa = torch.clamp(batch_kappa, max=50.0) # WAS 10.0
                else:
                    batch_kappa = torch.clamp(batch_kappa, max=100.0)

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
                    
                    # Update Mu
                    new_mu = self.alpha * old_mu + (1 - self.alpha) * mean_dir
                    new_mu = F.normalize(new_mu, p=2, dim=0)
                    
                    # Update Kappa
                    new_kappa = self.alpha * old_kappa + (1 - self.alpha) * batch_kappa
                    
                    self.prototypes[cval]['mu'] = new_mu
                    self.prototypes[cval]['kappa'] = new_kappa
                    self.prototypes[cval]['count'] += 1

        # Periodic Visualization
        if self.epoch_counter > 0 and self.epoch_counter % 10 == 0:
            self.save_tsne_plot()

    def query_voxelwise_novelty(self, embedding_3d: torch.Tensor):
        """
        Computes Free Energy for every voxel based on stored prototypes.
        """
        if not self.prototypes:
            # Fallback if memory is empty
            return torch.ones_like(embedding_3d[:,0]), torch.zeros_like(embedding_3d[:,0])

        B, F_dim, D, H, W = embedding_3d.shape
        
        # 1. Prepare Prototypes [C, F] and [C, 1]
        classes = sorted(self.prototypes.keys())
        mus = torch.stack([self.prototypes[c]['mu'] for c in classes]).to(embedding_3d.device)
        kappas = torch.stack([self.prototypes[c]['kappa'] for c in classes]).to(embedding_3d.device).unsqueeze(1)
        
        # 2. Flatten Image: [B, N, F]
        emb_flat = embedding_3d.view(B, F_dim, -1).permute(0, 2, 1)
        emb_flat = F.normalize(emb_flat, p=2, dim=2) 

        # 3. Compute Logits
        # Similarity: [B, N, F] @ [F, C] -> [B, N, C]
        cosine_sim = torch.matmul(emb_flat, mus.t())
        
        # Scale by Concentration (Depth of Well)
        # Broadcasting: [B, N, C] * [1, 1, C]
        logits = cosine_sim * kappas.view(1, 1, -1)
        
        # 4. Compute Free Energy E(z) = -LogSumExp(logits)
        # High Energy = Low Probability = Anomaly
        energy_flat = -torch.logsumexp(logits, dim=2) 
        
        # 5. Prediction (Deepest Well)
        val, idx = torch.max(logits, dim=2)
        pred_flat = torch.tensor(classes, device=embedding_3d.device)[idx]
        
        return energy_flat.view(B, D, H, W), pred_flat.view(B, D, H, W)

    def save_tsne_plot(self, perplexity=30.0, random_state=42):
        if len(self.prototypes) < 2: return

        # Extract data to CPU numpy
        class_ids = sorted(self.prototypes.keys())
        embeddings = torch.stack([self.prototypes[c]['mu'] for c in class_ids]).cpu().numpy()
        kappas = torch.stack([self.prototypes[c]['kappa'] for c in class_ids]).cpu().numpy()
        
        # Visual size proportional to Concentration
        sizes = 100 + (kappas - kappas.min()) / (kappas.max() - kappas.min() + 1e-6) * 200

        try:
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
            
            plot_filename = os.path.join(self.save_path, f"prototypes_epoch_{self.epoch_counter}.png")
            plt.savefig(plot_filename, dpi=300)
            plt.close()
        except Exception as e:
            print(f"[WARNING] t-SNE plot failed: {e}")