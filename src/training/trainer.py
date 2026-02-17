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
        scheduler=None
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
        
        # 1. Define Checkpoint Dir FIRST
        self.checkpoint_dir = config["training"]["checkpoint_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # 2. Initialize Logger
        self.logger = ExperimentLogger(
            log_dir=self.checkpoint_dir, 
            config=config,
            project_name="OpenWell_BTCV" 
        )
        
        # 3. Initialize Visualizer
        self.vis = TrainingVisualizer(save_dir=os.path.join(self.checkpoint_dir, "vis_debug"))

        # Mixed Precision
        self.scaler = torch.cuda.amp.GradScaler()
        
        # Metrics & Post-processing
        # FIX: reduction="mean_batch" allows us to see Dice for EACH CLASS separately
        self.dice_metric = DiceMetric(include_background=False, reduction="mean_batch", get_not_nans=False)
        
        self.post_label = AsDiscrete(to_onehot=config["model"]["out_channels"])
        self.post_pred = AsDiscrete(argmax=True, to_onehot=config["model"]["out_channels"])
        
        # State
        self.start_epoch = 1
        self.best_dice = 0.0
        self.global_step = 0
        
        # Configuration
        self.warmup_epochs = 800       #Disable Memory Bank for first X epochs
        self.full_vmf_epoch = 1000     # Reach max weight here (Slow fade-in)
        self.max_vmf_weight = 0.1

    def get_vmf_weight(self, epoch):
        """Linearly ramps up vMF weight to prevent shock."""
        if epoch <= self.warmup_epochs:
            return 0.0
        elif epoch >= self.full_vmf_epoch:
            return self.max_vmf_weight
        else:
            # Linear interpolation
            progress = (epoch - self.warmup_epochs) / (self.full_vmf_epoch - self.warmup_epochs)
            return self.max_vmf_weight * progress
        
    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0

        # Calculate current vMF weight
        vmf_weight = self.get_vmf_weight(epoch)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}(vMF: {vmf_weight:.4f})", dynamic_ncols=True)
        
        for step, batch in enumerate(pbar):
            self.global_step += 1
            inputs = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                logits, embedding = self.model(inputs)
                
                loss_seg = self.loss_fn(logits, labels)

                loss_vmf = torch.tensor(0.0, device=self.device)
                
                if self.memory_bank:
                    labels_sq = labels.squeeze(1)
                    embedding_f32 = embedding.detach().float() 
                    
                    self.memory_bank.update_prototypes(embedding_f32, labels_sq)
                    
                #Only compute loss if warmup is over
                if self.memory_bank and vmf_weight > 0:
                    loss_vmf = compute_vmf_loss(
                        embedding.float(), 
                        labels_sq, 
                        self.memory_bank, 
                        ignore_index=255
                    )
                # --------------------

                # Weight vMF loss (0.0 during warmup)
                loss = loss_seg + vmf_weight * loss_vmf

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.scheduler:
                self.scheduler.step()

            epoch_loss += loss.item()
            
            # Log Metrics
            metrics = {
                "Train/Step_Loss": loss.item(),
                "Train/Step_Seg": loss_seg.item(),
                "Train/Step_vMF": loss_vmf.item(),
                "Train/Step_LR": self.optimizer.param_groups[0]['lr']
            }
            self.logger.log_metrics(metrics, step=self.global_step)

            pbar.set_postfix({
                "Seg": f"{loss_seg.item():.4f}", 
                "vMF": f"{loss_vmf.item():.4f}"
            })

            # Snapshot every 100 steps
            if self.global_step % 200 == 0:
                self._log_visualization(inputs, labels, logits, embedding, epoch, step)

        return epoch_loss / len(self.train_loader)

    def validate(self, epoch):
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
                        predictor=self.model
                    )
                
                val_labels_list = decollate_batch(val_labels)
                val_outputs_list = decollate_batch(val_outputs)
                
                val_labels_convert = [
                    self.post_label(self._safe_map_ignore(label)) for label in val_labels_list
                ]
                val_outputs_convert = [
                    self.post_pred(pred) for pred in val_outputs_list
                ]
                
                self.dice_metric(y_pred=val_outputs_convert, y=val_labels_convert)

            # --- PER-CLASS DICE LOGIC ---
            # aggregate() returns [C] tensor because we set reduction="mean_batch"
            dice_scores_per_class = self.dice_metric.aggregate() 
            mean_dice = dice_scores_per_class.mean().item()
            
            # Log Mean Dice
            self.logger.log_scalar("Val/Dice_Mean", mean_dice, self.global_step)
            
            # Log Per-Class Dice & Print Debug Table
            print(f"\n{'='*40}")
            print(f"Epoch {epoch} Validation Report")
            print(f"{'='*40}")
            print(f"Mean Dice: {mean_dice:.4f}")
            print(f"{'-'*40}")
            
            for i, score in enumerate(dice_scores_per_class):
                class_score = score.item()
                # Log to WandB: "Val/Dice_Class_1", "Val/Dice_Class_2", etc.
                self.logger.log_scalar(f"Val/Dice_Class_{i+1}", class_score, self.global_step)
                print(f"Class {i+1:02d}: {class_score:.4f}")
            
            print(f"{'='*40}\n")
            
            self.dice_metric.reset()
            
        return mean_dice

    def _safe_map_ignore(self, label_tensor):
        """Maps 255 to 0 for validation Dice calculation only."""
        label_tensor[label_tensor == 255] = 0
        return label_tensor

    def _log_visualization(self, inputs, labels, logits, embedding, epoch, step):
        with torch.no_grad():
            if self.memory_bank:
                energy_map, _ = self.memory_bank.query_voxelwise_novelty(embedding.float())
            else:
                energy_map = torch.zeros_like(labels).squeeze(1).float()
            
            preds = torch.argmax(logits, dim=1)
            
            save_path = self.vis.log_snapshot(inputs, labels, preds, energy_map, epoch, step)
            
            if save_path:
                self.logger.log_image("Debug/Snapshot", save_path, self.global_step)

    def save_checkpoint(self, epoch, metric, is_best=False):
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
            self.memory_bank.save_memory_bank(
                os.path.join(self.checkpoint_dir, "energy_memory_bank.pth")
            )

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_checkpoint.pth")
            torch.save(state, best_path)
            print(f"[INFO] New Best Model Saved! (Dice: {metric:.4f})")

    def load_checkpoint(self, path):
        if not os.path.exists(path):
            print(f"[WARNING] Checkpoint not found at {path}. Starting from scratch.")
            return

        print(f"[INFO] Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_dice = checkpoint.get("best_dice", 0.0)
        self.global_step = checkpoint.get("global_step", 0)
        
        if self.memory_bank:
            mb_path = os.path.join(os.path.dirname(path), "energy_memory_bank.pth")
            self.memory_bank.load_memory_bank(mb_path, self.device)

    def fit(self):
        max_epochs = self.config["training"]["max_iterations"] // len(self.train_loader)
        eval_freq = self.config["training"]["eval_num"]
        
        # Resume if needed
        if self.config["training"].get("resume", False):
            self.load_checkpoint(os.path.join(self.checkpoint_dir, "best_checkpoint.pth"))

        print(f"[INFO] Starting training: Epoch {self.start_epoch} to {max_epochs}")

        for epoch in range(self.start_epoch, max_epochs + 1):
            if self.memory_bank:
                self.memory_bank.epoch_counter = epoch

            train_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch} | Train Loss: {train_loss:.5f}")

            if epoch % eval_freq == 0:
                val_dice = self.validate(epoch)
                
                is_best = val_dice > self.best_dice
                if is_best:
                    self.best_dice = val_dice
                
                self.save_checkpoint(epoch, val_dice, is_best)

        print(f"[INFO] Training Completed. Best Dice: {self.best_dice:.4f}")
        self.logger.close()