import os
import torch
import torch.nn as nn
from tqdm import tqdm

from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.data import decollate_batch

from src.training.losses import compute_vmf_loss
from src.utils.visualizer import TrainingVisualizer
from src.utils.logger import ExperimentLogger


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        memory_bank,
        device,
        config,
        scheduler=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.memory_bank = memory_bank
        self.device = device
        self.config = config
        self.scheduler = scheduler

        # ============================================================
        # Class weights (BTCV: 0..10) for CrossEntropy inside hybrid loss
        # ============================================================
        class_weights = torch.tensor(
            [
                0.1,   # 0: Background
                1.0,   # 1: Spleen
                1.0,   # 2: R. Kidney
                1.0,   # 3: L. Kidney
                10.0,  # 4: Gallbladder (BOOST)
                10.0,  # 5: Esophagus   (BOOST)
                0.5,   # 6: Liver       (SUPPRESS)
                1.0,   # 7: Stomach
                1.0,   # 8: Aorta
                1.0,   # 9: IVC
                10.0,  # 10: Veins      (BOOST)
            ],
            dtype=torch.float32,
            device=device,
        )

        # Inject weights into the CrossEntropy part of your hybrid loss
        self.loss_fn.ce = nn.CrossEntropyLoss(ignore_index=255, weight=class_weights)
        # ============================================================

        # Checkpoints
        self.checkpoint_dir = config["training"]["checkpoint_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Logger + Visualizer
        self.logger = ExperimentLogger(
            log_dir=self.checkpoint_dir,
            config=config,
            project_name="OpenWell_BTCV",
        )
        self.vis = TrainingVisualizer(save_dir=os.path.join(self.checkpoint_dir, "vis_debug"))

        # Mixed Precision
        self.scaler = torch.cuda.amp.GradScaler()

        # Metrics & Post-processing
        self.dice_metric = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=False)
        self.post_label = AsDiscrete(to_onehot=config["model"]["out_channels"])
        self.post_pred = AsDiscrete(argmax=True, to_onehot=config["model"]["out_channels"])

        # State
        self.start_epoch = 1
        self.best_dice = 0.0
        self.global_step = 0

        # ----------------------------
        # vMF / Energy settings
        # ----------------------------
        vmf_cfg = config.get("training", {}).get("vmf", {})

        # IMPORTANT: defaults below should fit small runs too
        self.vmf_warmup_steps = int(vmf_cfg.get("warmup_steps", 800))
        self.vmf_ramp_steps = int(vmf_cfg.get("ramp_steps", 2200))
        self.max_vmf_weight = float(vmf_cfg.get("max_weight", 0.05))

        self.vmf_tau = float(vmf_cfg.get("tau", 1.0))
        self.vmf_reject_weight = float(vmf_cfg.get("reject_weight", 0.2))
        self.vmf_include_background = bool(vmf_cfg.get("include_background", False))
        
        # Optional grad clipping
        self.grad_clip_norm = vmf_cfg.get("grad_clip_norm", None)
        if self.grad_clip_norm is not None:
            self.grad_clip_norm = float(self.grad_clip_norm)

        # Optional step-based validation
        self.eval_every_steps = config.get("training", {}).get("eval_every_steps", None)
        if self.eval_every_steps is not None:
            self.eval_every_steps = int(self.eval_every_steps)

    def get_vmf_weight(self, step: int) -> float:
        """Linear ramp in steps: 0 until warmup, then ramp to max_vmf_weight."""
        if step < self.vmf_warmup_steps:
            return 0.0
        t = (step - self.vmf_warmup_steps) / max(1, self.vmf_ramp_steps)
        t = float(min(1.0, max(0.0, t)))
        return self.max_vmf_weight * t

    def train_epoch(self, epoch: int):
        self.model.train()
        epoch_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", dynamic_ncols=True)

        for step, batch in enumerate(pbar):
            self.global_step += 1

            # step-based vMF weight
            vmf_weight = self.get_vmf_weight(self.global_step)

            inputs = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            labels_sq = labels.squeeze(1)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                logits, embedding = self.model(inputs)
                loss_seg = self.loss_fn(logits, labels)

                loss_vmf = torch.tensor(0.0, device=self.device)

                # === FIX: ALWAYS UPDATE MEMORY (Silent Warmup) ===
                # We removed the 'can_update_memory' check. 
                # Updates happen every step so prototypes are ready for step 800.
                if self.memory_bank:
                    self.memory_bank.update_prototypes(embedding.detach().float(), labels_sq)

                # Compute vMF Loss ONLY after warmup
                if self.memory_bank and vmf_weight > 0.0:
                    loss_vmf = compute_vmf_loss(
                        embedding.float(),
                        labels_sq,
                        self.memory_bank,
                        tau=self.vmf_tau,
                        ignore_index=255,
                        reject_weight=self.vmf_reject_weight,
                        include_background=self.vmf_include_background,
                    )

                loss = loss_seg + vmf_weight * loss_vmf

            # Backward + step (AMP)
            self.scaler.scale(loss).backward()

            # Optional grad clipping
            if self.grad_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler:
                self.scheduler.step()

            epoch_loss += float(loss.item())

            # Logging
            metrics = {
                "Train/Step_Loss": float(loss.item()),
                "Train/Step_Seg": float(loss_seg.item()),
                "Train/Step_vMF": float(loss_vmf.item()),
                "Train/Step_LR": float(self.optimizer.param_groups[0]["lr"]),
                "Train/vMF_weight": float(vmf_weight),
            }
            self.logger.log_metrics(metrics, step=self.global_step)

            pbar.set_postfix(
                {
                    "Seg": f"{loss_seg.item():.4f}",
                    "vMF": f"{loss_vmf.item():.4f}",
                    "w": f"{vmf_weight:.4f}",
                }
            )

            # Snapshot
            if self.global_step % 200 == 0:
                self._log_visualization(inputs, labels, logits, embedding, epoch, step)

            # Optional: step-based validation
            if self.eval_every_steps is not None and (self.global_step % self.eval_every_steps == 0):
                val_dice = self.validate(epoch)
                is_best = val_dice > self.best_dice
                if is_best:
                    self.best_dice = val_dice
                self.save_checkpoint(epoch, val_dice, is_best)

        return epoch_loss / max(1, len(self.train_loader))

    def validate(self, epoch: int):
        self.model.eval()
        print(f"[INFO] Starting Validation for Epoch {epoch}...")

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                val_inputs = batch["image"].to(self.device)
                val_labels = batch["label"].to(self.device)

                with torch.cuda.amp.autocast():
                    val_outputs, _ = sliding_window_inference(
                        val_inputs,
                        roi_size=(96, 96, 96),
                        sw_batch_size=4,
                        overlap=0.5,
                        predictor=self.model,
                    )

                val_labels_list = decollate_batch(val_labels)
                val_outputs_list = decollate_batch(val_outputs)

                val_labels_convert = [self.post_label(self._safe_map_ignore(lbl)) for lbl in val_labels_list]
                val_outputs_convert = [self.post_pred(pred) for pred in val_outputs_list]

                self.dice_metric(y_pred=val_outputs_convert, y=val_labels_convert)

            dice_scores_per_class = self.dice_metric.aggregate()
            mean_dice = dice_scores_per_class.mean().item()

            self.logger.log_scalar("Val/Dice_Mean", mean_dice, self.global_step)

            print(f"\n{'='*40}")
            print(f"Epoch {epoch} Validation Report")
            print(f"{'='*40}")
            print(f"Mean Dice: {mean_dice:.4f}")
            print(f"{'-'*40}")

            for i, score in enumerate(dice_scores_per_class):
                class_score = score.item()
                self.logger.log_scalar(f"Val/Dice_Class_{i+1}", class_score, self.global_step)
                print(f"Class {i+1:02d}: {class_score:.4f}")

            print(f"{'='*40}\n")

            self.dice_metric.reset()

        return mean_dice

    def _safe_map_ignore(self, label_tensor):
        """Maps 255 -> 0 for validation Dice calculation only."""
        label_tensor = label_tensor.clone()
        label_tensor[label_tensor == 255] = 0
        return label_tensor

    def _log_visualization(self, inputs, labels, logits, embedding, epoch, step):
            with torch.no_grad():
                # 1. Get Segmentation Probabilities (Softmax)
                probs = torch.softmax(logits, dim=1)
                prob_bg = probs[:, 0, :, :, :]
                
                # 2. Calculate the "Organ Gate" (1 = Foreground, 0 = Background)
                gate = 1.0 - prob_bg
                
                # 3. Get Raw Energy
                if self.memory_bank:
                    # include_background=False creates "Organ-Relative Energy"
                    raw_energy, _ = self.memory_bank.query_voxelwise_novelty(
                        embedding.float(), 
                        tau=self.vmf_tau, 
                        include_background=self.vmf_include_background
                    )
                else:
                    raw_energy = torch.zeros_like(labels).squeeze(1).float()
                
                preds = torch.argmax(logits, dim=1)
                
                # 4. Pass EVERYTHING to the visualizer (Do NOT multiply them here)
                # Make sure your visualizer.log_snapshot accepts raw_energy and gate separately!
                save_path = self.vis.log_snapshot(
                    inputs=inputs, 
                    labels=labels, 
                    preds=preds, 
                    raw_energy=raw_energy,  # Pass raw, unnormalized energy
                    prob_bg=prob_bg,        # Pass the raw background probability
                    epoch=epoch, 
                    step=step
                )
                
                if save_path:
                    self.logger.log_image("Debug/Snapshot", save_path, self.global_step)

    def save_checkpoint(self, epoch: int, metric: float, is_best: bool = False):
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_dice": metric,
            "global_step": self.global_step,
        }

        filename = os.path.join(self.checkpoint_dir, "checkpoint_last.pth")
        torch.save(state, filename)

        if self.memory_bank:
            self.memory_bank.save_memory_bank(os.path.join(self.checkpoint_dir, "energy_memory_bank.pth"))

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_checkpoint.pth")
            torch.save(state, best_path)
            print(f"[INFO] New Best Model Saved! (Dice: {metric:.4f})")

    def load_checkpoint(self, path: str):
        if not os.path.exists(path):
            print(f"[WARNING] Checkpoint not found at {path}. Starting from scratch.")
            return

        print(f"[INFO] Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.best_dice = float(checkpoint.get("best_dice", 0.0))
        self.global_step = int(checkpoint.get("global_step", 0))

        if self.memory_bank:
            mb_path = os.path.join(os.path.dirname(path), "energy_memory_bank.pth")
            self.memory_bank.load_memory_bank(mb_path, self.device)

    def fit(self):
        max_epochs = self.config["training"]["max_iterations"] // max(1, len(self.train_loader))
        eval_freq_epochs = self.config["training"].get("eval_num", None)

        # Resume if needed
        if self.config["training"].get("resume", False):
            # Point this to your BEST checkpoint (recovery_checkpoint.pth)
            self.load_checkpoint(os.path.join(self.checkpoint_dir, "best_checkpoint.pth"))
            # Adjust max_epochs if you want to extend training
            
        print(f"[INFO] Starting training: Epoch {self.start_epoch} to {max_epochs}")

        for epoch in range(self.start_epoch, max_epochs + 1):
            if self.memory_bank:
                self.memory_bank.epoch_counter = epoch

            train_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch} | Train Loss: {train_loss:.5f}")

            if eval_freq_epochs is not None and self.eval_every_steps is None:
                if epoch % int(eval_freq_epochs) == 0:
                    val_dice = self.validate(epoch)
                    is_best = val_dice > self.best_dice
                    if is_best:
                        self.best_dice = val_dice
                    self.save_checkpoint(epoch, val_dice, is_best)

        print(f"[INFO] Training Completed. Best Dice: {self.best_dice:.4f}")
        self.logger.close()

