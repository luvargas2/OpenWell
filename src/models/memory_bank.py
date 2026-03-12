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

        # --- Fix 3: Adaptive Lambda ---
        # Running estimate of the 95th percentile of in-distribution free energies.
        # Used as the threshold λ in ReLU(F(z) - λ) for the novelty map.
        # Initialised to 0.0; updates via EMA once training starts.
        self.adaptive_lambda = 0.0
        self._lambda_ema_alpha = 0.95  # Heavy smoothing; λ should be stable

    def save_memory_bank(self, save_path):
        if not self.prototypes:
            return
        save_dict = {}
        for k, v in self.prototypes.items():
            save_dict[k] = {
                'mu':    v['mu'].detach().cpu(),
                'kappa': v['kappa'].detach().cpu(),
                'R_bar': v.get('R_bar', torch.tensor(0.0)).detach().cpu(),
                'count': v['count'],
            }
        # Persist calibrated lambda so inference does not need to recompute it
        save_dict['__meta__'] = {'adaptive_lambda': self.adaptive_lambda}
        torch.save(save_dict, save_path)
        print(f"[INFO] Prototypes saved to {save_path}  (adaptive_lambda={self.adaptive_lambda:.4f})")

    def load_memory_bank(self, memory_bank_path, device="cuda"):
        if not os.path.exists(memory_bank_path):
            print(f"[WARNING] Memory bank file not found: {memory_bank_path}. Starting fresh.")
            return

        checkpoint = torch.load(memory_bank_path, map_location=device, weights_only=False)

        # Restore calibrated lambda if present
        meta = checkpoint.pop('__meta__', {})
        if 'adaptive_lambda' in meta:
            self.adaptive_lambda = float(meta['adaptive_lambda'])
            print(f"[INFO] Restored adaptive_lambda={self.adaptive_lambda:.4f} from checkpoint")

        self.prototypes = {}
        for k, v in checkpoint.items():
            self.prototypes[k] = {
                'mu':    v['mu'].to(device),
                'kappa': v['kappa'].to(device),
                'R_bar': v.get('R_bar', torch.tensor(0.0)).to(device),
                'count': v['count'],
            }
        print(f"[INFO] Loaded {len(self.prototypes)} classes into Memory Bank.")

    # ==========================================
    # CORE LEARNING (Phase 1: Training)
    # ==========================================
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
                
                # vMF MLE approximation for Kappa (Banerjee et al. 2005, Eq. 4 in paper):
                # κ = (R̄·F − R̄³) / (1 − R̄²)
                batch_kappa = (R * self.feature_dim - R**3) / (1 - R**2 + 1e-6)
                
                # --- Fix 2 + Fix 5: Kappa clamping ---
                # Background: shallow well (kappa ≤ 10) so it cannot "swallow"
                # unseen structures in the energy landscape.
                # Organ classes: allow up to 100 so tight clusters (spleen, aorta)
                # form genuinely deep wells. The kappa-normalisation in the loss
                # (Fix 4) prevents gradient vanishing from high-κ dominance.
                if cval == 0:
                    batch_kappa = torch.clamp(batch_kappa, min=0.5, max=10.0)
                else:
                    # Raised from 100 → 500: R_bar≈0.95-0.98 produces raw κ≈1000-2000+,
                    # previously all clamped identically to 100. Allowing up to 500 lets
                    # tighter clusters (spleen, aorta) form deeper wells than looser ones
                    # (gallbladder, esophagus), increasing energy separation for OOD voxels.
                    batch_kappa = torch.clamp(batch_kappa, min=1.0, max=500.0)

                # 4. Update Memory (EMA)
                if cval not in self.prototypes:
                    self.prototypes[cval] = {
                        'mu': mean_dir.detach(),
                        'kappa': batch_kappa.detach(),
                        'R_bar': R.detach(),   # store mean-vector length for diagnostics
                        'count': 1
                    }
                else:
                    old_mu = self.prototypes[cval]['mu']
                    old_kappa = self.prototypes[cval]['kappa']

                    # Update Mu
                    new_mu = self.alpha * old_mu + (1 - self.alpha) * mean_dir
                    new_mu = F.normalize(new_mu, p=2, dim=0)

                    # Update Kappa and R_bar
                    new_kappa = self.alpha * old_kappa + (1 - self.alpha) * batch_kappa
                    old_R = self.prototypes[cval].get('R_bar', R.detach())
                    new_R = self.alpha * old_R + (1 - self.alpha) * R.detach()

                    self.prototypes[cval]['mu'] = new_mu
                    self.prototypes[cval]['kappa'] = new_kappa
                    self.prototypes[cval]['R_bar'] = new_R
                    self.prototypes[cval]['count'] += 1

        # Periodic Visualization
        if self.epoch_counter > 0 and self.epoch_counter % 10 == 0:
            self.save_tsne_plot()

    # ==========================================
    # INTERACTIVE ENROLLMENT (Phase 2: Inference)
    # ==========================================
    @torch.no_grad()
    def enroll_interactive_prototype(self, new_class_id: int, mu_new: torch.Tensor, kappa_new: torch.Tensor):
        """
        Instantly adds a new class to the memory bank based on user guidance (Section 3.5).
        This allows the model to immediately recognize this class in subsequent scans 
        without any gradient-based retraining.
        """
        # Ensure tensors are normalized and on the correct device
        mu_new = F.normalize(mu_new.detach(), p=2, dim=0)
        kappa_new = kappa_new.detach()
        
        # Add to the dictionary. We set count to 1 so it can be updated via EMA later if desired.
        self.prototypes[new_class_id] = {
            'mu': mu_new,
            'kappa': kappa_new,
            'count': 1
        }
        print(f"[INFO] Successfully enrolled New Class {new_class_id} into Memory Bank!")
        print(f"       -> Concentration (Kappa): {kappa_new.item():.2f}")

    # ==========================================
    # FIX 3: ADAPTIVE LAMBDA
    # ==========================================
    @torch.no_grad()
    def update_adaptive_lambda(self, energy_map: torch.Tensor, labels: torch.Tensor):
        """
        Update the adaptive energy threshold λ using in-distribution voxels.

        λ = EMA( p95( F(z) | z belongs to a known organ class ) )

        Reasoning: if the 95th percentile of known-class free energies is λ,
        then ReLU(F(z) - λ) will fire only for voxels that are more novel than
        95% of known-class voxels — a principled, calibrated noise floor.
        """
        labels_flat = labels.reshape(-1)
        energy_flat = energy_map.reshape(-1).float()

        # Known-class mask: exclude background (0), ignore label (255), and unknown (999)
        known_mask = (labels_flat > 0) & (labels_flat != self.IGNORE_LABEL) & (labels_flat != self.UNKNOWN_CLASS_ID)
        n_known = known_mask.sum().item()
        if n_known < 50:
            return  # Not enough voxels for a stable estimate

        known_energies = energy_flat[known_mask]
        p95 = torch.quantile(known_energies, 0.95).item()

        # EMA update
        if self.adaptive_lambda == 0.0:
            self.adaptive_lambda = p95  # Cold start: use first estimate directly
        else:
            self.adaptive_lambda = (self._lambda_ema_alpha * self.adaptive_lambda
                                    + (1 - self._lambda_ema_alpha) * p95)

    # ==========================================
    # INFERENCE & ENERGY CALCULATION
    # ==========================================
    def query_voxelwise_novelty(self, embedding_3d: torch.Tensor, tau: float = 1.0, include_background: bool = False):
        """
        Computes Free Energy for every voxel based on stored prototypes (Eq. 7).
        """
        if not self.prototypes:
            return torch.ones_like(embedding_3d[:,0]), torch.zeros_like(embedding_3d[:,0])

        B, F_dim, D, H, W = embedding_3d.shape
        
        classes = sorted(self.prototypes.keys())
        
        if not include_background:
             classes = [c for c in classes if int(c) != 0]
        
        if not classes:
            return torch.zeros_like(embedding_3d[:,0]), torch.zeros_like(embedding_3d[:,0])

        mus = torch.stack([self.prototypes[c]['mu'] for c in classes]).to(embedding_3d.device)
        kappas = torch.stack([self.prototypes[c]['kappa'] for c in classes]).to(embedding_3d.device)

        # --- Fix 4 (inference): normalise kappas by their mean ---
        # Prevents high-κ classes (e.g. spleen) from monopolising the energy landscape.
        kappa_mean = kappas.mean().clamp(min=1.0)
        kappas_norm = kappas / kappa_mean  # shape [C]

        emb_flat = embedding_3d.view(B, F_dim, -1).permute(0, 2, 1)
        emb_flat = F.normalize(emb_flat, p=2, dim=2)

        cosine_sim = torch.matmul(emb_flat, mus.t())  # [B, N, C]

        # Scale by normalised concentration and temperature
        logits = (cosine_sim * kappas_norm.view(1, 1, -1)) / tau

        # Free Energy E(z) = -LogSumExp(logits)
        energy_flat = -torch.logsumexp(logits, dim=2)  # [B, N]

        val, idx = torch.max(logits, dim=2)
        pred_flat = torch.tensor(classes, device=embedding_3d.device)[idx]

        return energy_flat.view(B, D, H, W), pred_flat.view(B, D, H, W)

    # ==========================================
    # VISUALIZATION
    # ==========================================
    def save_tsne_plot(self, perplexity=30.0, random_state=42):
        if len(self.prototypes) < 2: return

        class_ids = sorted(self.prototypes.keys())
        embeddings = torch.stack([self.prototypes[c]['mu'] for c in class_ids]).cpu().numpy()
        kappas = torch.stack([self.prototypes[c]['kappa'] for c in class_ids]).cpu().numpy()
        
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