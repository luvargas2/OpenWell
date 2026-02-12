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

        # 2. Initialize Logger (Now safe)
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
        # include_background=False ignores class 0.
        # We will map '255' -> '0' in validation so it gets ignored too.
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
        self.post_label = AsDiscrete(to_onehot=config["model"]["out_channels"])
        self.post_pred = AsDiscrete(argmax=True, to_onehot=config["model"]["out_channels"])
        
        # State
        self.start_epoch = 1
        self.best_dice = 0.0
        self.global_step = 0

    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0
        epoch_seg_loss = 0
        epoch_vmf_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", dynamic_ncols=True)
        
        for step, batch in enumerate(pbar):
            self.global_step += 1
            inputs = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                logits, embedding = self.model(inputs)
                
                # Main Loss (Handles 255 internally)
                loss_seg = self.loss_fn(logits, labels)
                
                # vMF Loss
                loss_vmf = torch.tensor(0.0, device=self.device)
                if self.memory_bank:
                    labels_sq = labels.squeeze(1)
                    # Update (No grad, skips 255)
                    self.memory_bank.update_prototypes(embedding.detach(), labels_sq)
                    # Loss (Skips 255)
                    loss_vmf = compute_vmf_loss(embedding, labels_sq, self.memory_bank, ignore_index=255)
                
                loss = loss_seg + 0.1 * loss_vmf

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.scheduler:
                self.scheduler.step()

            # Logs
            current_lr = self.optimizer.param_groups[0]['lr']
            epoch_loss += loss.item()
            epoch_seg_loss += loss_seg.item()
            epoch_vmf_loss += loss_vmf.item()
            
            self.logger.log_scalar("Train/Step_Loss", loss.item(), self.global_step)
            self.logger.log_scalar("Train/Step_LR", current_lr, self.global_step)

            pbar.set_postfix({
                "Seg": f"{loss_seg.item():.4f}", 
                "vMF": f"{loss_vmf.item():.4f}"
            })

            # Snapshot every 100 steps
            if self.global_step % 5 == 0:
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
                
                # Map 255 -> 0 before OneHot to avoid crash
                val_labels_convert = [
                    self.post_label(self._safe_map_ignore(label)) for label in val_labels_list
                ]
                val_outputs_convert = [
                    self.post_pred(pred) for pred in val_outputs_list
                ]
                
                self.dice_metric(y_pred=val_outputs_convert, y=val_labels_convert)

            mean_dice = self.dice_metric.aggregate().item()
            self.dice_metric.reset()
            
        print(f"Epoch {epoch} | Validation Dice: {mean_dice:.4f}")
        self.logger.log_scalar("Val/Dice", mean_dice, epoch)
        return mean_dice

    def _safe_map_ignore(self, label_tensor):
        """Maps 255 (Ignore) to 0 (Background) for Validation metrics."""
        label_tensor[label_tensor == 255] = 0
        return label_tensor

    def _log_visualization(self, inputs, labels, logits, embedding, epoch, step):
        with torch.no_grad():
            if self.memory_bank:
                energy_map, _ = self.memory_bank.query_voxelwise_novelty(embedding)
            else:
                energy_map = torch.zeros_like(labels).squeeze(1).float()
            
            preds = torch.argmax(logits, dim=1)
            
            # Save Locally
            save_path = self.vis.log_snapshot(inputs, labels, preds, energy_map, epoch, step)
            
            # Log to WandB
            if save_path:
                self.logger.log_image("Debug/Snapshot", save_path, step)

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

    def fit(self):
        max_epochs = self.config["training"]["max_iterations"] // len(self.train_loader)
        eval_freq = self.config["training"]["eval_num"]
        
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