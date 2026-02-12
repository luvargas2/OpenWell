# import os
# import torch
# from monai.inferers import sliding_window_inference
# from monai.metrics import DiceMetric
# from monai.transforms import AsDiscrete
# from monai.data import decollate_batch
# from tqdm import tqdm

# from openwell.src.models.memory_bank import MemoryBankV
# from losses.proto_loss import compute_vmf_loss
# from preprocess.brats import extract_one_shot_unknown_sample

# def save_feature_embeddings(features, labels, epoch, save_dir="./feature_logs"):
#     os.makedirs(save_dir, exist_ok=True)
#     save_path = os.path.join(save_dir, f"features.pt")
#     torch.save({"features": features.cpu(), "labels": labels.cpu()}, save_path)
#     print(f" Feature embeddings saved at epoch {epoch} -> {save_path}")

# def validation(model, val_loader, dice_metric, device, post_label, post_pred):
#     model.eval()
#     with torch.no_grad():
#         for batch in val_loader:
#             val_inputs, val_labels = batch["image"].to(device), batch["label"].to(device)
#             with torch.amp.autocast("cuda"):
#                 val_outputs,_ = sliding_window_inference(
#                     val_inputs, roi_size=(96, 96, 96), sw_batch_size=1, overlap=0.5, predictor=model
#                 )
#             val_labels_list = decollate_batch(val_labels)
#             val_outputs_list = decollate_batch(val_outputs)
#             val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
#             val_outputs_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
#             dice_metric(y_pred=val_outputs_convert, y=val_labels_convert)

#         mean_dice = dice_metric.aggregate().item()
#         dice_metric.reset()
#     torch.cuda.empty_cache()
#     return mean_dice

# def train_model(
#     model, train_loader, val_loader, test_loader, config, loss_function, optimizer, scaler, checkpoint_dir,  device, scheduler=None
# ):
#     if checkpoint_dir is None:
#         print("[WARNING] checkpoint_dir was None in config. Defaulting to './outputs'.")
#         checkpoint_dir = "./outputs"
#     os.makedirs(checkpoint_dir, exist_ok=True)

#     max_iterations = config["training"]["max_iterations"]
#     eval_num = config["training"]["eval_num"]
#     use_memory_bank = config["training"].get("use_memory_bank", False)
#     resume = config["training"].get("resume", False)
#     embed_dim = config["model"].get("embed_dim_final", 128)
#     memory_bank_path = os.path.join(checkpoint_dir, "energy_memory_bank.pth")

#     memory_bank = None
#     if use_memory_bank:
#         memory_bank = MemoryBankV(
#             memory_size=config["training"]["memory_size"],
#             feature_dim=embed_dim,
#             alpha=0.99,
#             save_path=os.path.join(checkpoint_dir,f"prototypes_{checkpoint_dir.split('/')[-1]}")
#         ).to(device)
#         print(f"[INFO] Energy-Based Memory Bank initialized with embed_dim={embed_dim}.")

#     post_label = AsDiscrete(to_onehot=config["model"]["out_channels"])
#     post_pred = AsDiscrete(argmax=True, to_onehot=config["model"]["out_channels"])
#     dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

#     start_epoch = 1
#     global_step = 0
#     dice_val_best = 0.0
#     global_step_best = 0
#     best_epoch=0
#     UNKNOWN_CLASS_ID=999

#     if resume:
#         resume_checkpoint = os.path.join(checkpoint_dir, "best_checkpoint.pth")
#         if os.path.exists(resume_checkpoint):
#             print(f"[INFO] Resuming from checkpoint: {resume_checkpoint}")
#             # weights_only=False is CRITICAL for loading checkpoints with custom objects
#             checkpoint_data = torch.load(resume_checkpoint, map_location=device, weights_only=False)
            
#             model.load_state_dict(checkpoint_data["model_state_dict"])
#             optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
            
#             if "scaler_state_dict" in checkpoint_data and checkpoint_data["scaler_state_dict"] is not None:
#                 scaler.load_state_dict(checkpoint_data["scaler_state_dict"])

#             if scheduler is not None and "scheduler_state_dict" in checkpoint_data and checkpoint_data["scheduler_state_dict"] is not None:
#                 scheduler.load_state_dict(checkpoint_data["scheduler_state_dict"])
#                 print("[INFO] Scheduler state loaded.")

#             start_epoch = checkpoint_data["epoch"] + 1
#             global_step = checkpoint_data.get("global_step", start_epoch * len(train_loader))
#             dice_val_best = checkpoint_data.get("dice_val_best", 0.0)
#             global_step_best = checkpoint_data.get("global_step_best",  start_epoch * len(train_loader))
#             best_epoch = start_epoch
            
#             if use_memory_bank and os.path.exists(memory_bank_path):
#                 memory_bank.load_memory_bank(memory_bank_path, device=device)
#                 print("[INFO] Memory bank reloaded for resume.")
#         else:
#             print(f"[WARNING] Resume is True, but {resume_checkpoint} not found. Starting training from scratch.")

#     one_shot_sample = extract_one_shot_unknown_sample(test_loader)
#     if one_shot_sample is not None and use_memory_bank:
#         print("[INFO] Using One-Shot Unknown Example for Registration")
#         model.eval()  
#         with torch.no_grad():
#             inputs, labels = one_shot_sample["image"].to(device), one_shot_sample["label"].to(device)
#             _, embedding= sliding_window_inference(inputs, roi_size=(96, 96, 96), sw_batch_size=4, overlap=0.5,
#                 mode="gaussian",predictor=model)
#             labels[labels > 1] = UNKNOWN_CLASS_ID 
#             labels = labels.squeeze(1) 
#             memory_bank.update_prototypes(embedding, labels)
            

#     num_epochs = max_iterations // len(train_loader) + 1 
#     print(f"[INFO] Starting training from epoch={start_epoch} to {num_epochs}")
    
#     global_iterator = tqdm(total=max_iterations, desc="Total Progress", dynamic_ncols=True)
#     global_iterator.update(global_step)
    
#     for epoch in range(start_epoch, num_epochs + 1):
#         model.train()
        
#         # --- ROBUST INITIALIZATION ---
#         epoch_loss = 0
#         epoch_seg_loss = 0.0
#         epoch_vmf_loss = 0.0 
#         step = 0 # Prevent UnboundLocalError if loop is empty
#         # -----------------------------
        
#         all_embeddings = []
#         all_labels = []

#         if use_memory_bank:
#             memory_bank.epoch_counter = epoch  
        
#         epoch_iterator = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", dynamic_ncols=True)

#         for step, batch in enumerate(epoch_iterator):
#             inputs, labels = batch["image"].to(device), batch["label"].to(device)
#             optimizer.zero_grad()
#             with torch.amp.autocast("cuda"):
#                 logits, embedding = model(inputs)
#                 seg_loss = loss_function(logits, labels)
#                 vmf_loss_val = 0.0
                
#                 if use_memory_bank:
#                     labels_sq = batch["label"].to(device).squeeze(1)
#                     memory_bank.update_prototypes(embedding.detach(), labels_sq)
                    
#                     #labels_sq[labels_sq > 1] = UNKNOWN_CLASS_ID
#                     vmf_loss = compute_vmf_loss(embedding, labels_sq, memory_bank, ignore_index=255)
#                     vmf_loss_val = vmf_loss.item()
                    
#                     loss = seg_loss + 0.1 * vmf_loss
#                 else:
#                     loss = seg_loss
                
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()

#             if scheduler is not None:
#                 scheduler.step()

#             epoch_loss += loss.item()
#             seg_loss_val = seg_loss.item()
#             epoch_seg_loss += seg_loss_val
#             epoch_vmf_loss += vmf_loss_val
 
#             global_step += 1
#             global_iterator.update(1)

#             postfix_dict = {
#                 "seg": f"{seg_loss_val:.4f}",
#                 "vmf": f"{vmf_loss_val:.4f}",
#                 "tot": f"{loss.item():.4f}",
#             }
#             if scheduler is not None:
#                 postfix_dict["lr"] = f"{scheduler.get_last_lr()[0]:.6f}"
#             epoch_iterator.set_postfix(postfix_dict)

#             if step % 5 == 0:
#                 all_embeddings.append(embedding.detach().cpu())
#                 all_labels.append(labels.detach().cpu())
            
#         os.makedirs(checkpoint_dir, exist_ok=True)
#         # Validation and checkpoint saving
#         if (epoch % 100 == 0 and global_step != 0) or global_step == max_iterations:
#             print("Saving embeddings...")
#             if all_embeddings:
#                 all_embeddings_tensor = torch.cat(all_embeddings, dim=0)
#                 all_labels_tensor = torch.cat(all_labels, dim=0)
#                 save_feature_embeddings(all_embeddings_tensor, all_labels_tensor, epoch, save_dir=os.path.join(checkpoint_dir,'feature_logs'))

#         if epoch == 1 or (epoch % eval_num == 0 and global_step != 0) or global_step == max_iterations:
#             torch.cuda.empty_cache() 
#             print("Starting validation...")
#             mean_dice = validation(
#                 model, val_loader, dice_metric, device, post_label, post_pred
#             )
            
#             if use_memory_bank and len(memory_bank.prototypes) > 0:
#                 memory_bank_path = os.path.join(checkpoint_dir, "energy_memory_bank.pth")
#                 memory_bank.save_memory_bank(memory_bank_path)
#                 print('Memory Bank saved!')

#             if mean_dice > dice_val_best:
#                 dice_val_best = mean_dice
#                 global_step_best = global_step
#                 best_epoch = epoch
 
#                 torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
#                 'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
#                 "global_step": global_step,
#                 "dice_val_best": dice_val_best,
#                 "global_step_best": global_step_best,
#                 }, os.path.join(checkpoint_dir, "best_checkpoint.pth"))
                
#                 print(f"Model saved! Best Dice: {dice_val_best:.4f}")
#             else:
#                 print(f"Model not saved. Best Dice: {dice_val_best:.4f}, Curr: {mean_dice:.4f}")

#             if global_step >= max_iterations:
#                 break

#         # Avoid division by zero if batch count is small/zero
#         print(f"Epoch {epoch} completed. Loss: {epoch_loss / (step + 1):.5f}")
#         epoch += 1 

#     print(f"Training completed! Best Dice: {dice_val_best:.4f}")import torch
import torch.nn as nn
from tqdm import tqdm
from monai.metrics import DiceMetric
from monai.data import decollate_batch
from src.utils.visualizer import TrainingVisualizer
import torch

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, memory_bank, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.memory_bank = memory_bank
        self.device = device
        self.config = config
        
        self.scaler = torch.cuda.amp.GradScaler()
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        
        # Observability
        self.vis = TrainingVisualizer(save_dir=os.path.join(config["training"]["checkpoint_dir"], "vis_debug"))
        self.best_dice = 0.0

    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for step, batch in enumerate(pbar):
            inputs = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                # 1. Forward
                logits, embedding = self.model(inputs)
                
                # 2. Main Segmentation Loss (Ignore 255 is handled inside loss_fn)
                loss_seg = self.loss_fn(logits, labels)
                
                # 3. Memory Bank & Energy
                loss_vmf = 0.0
                if self.memory_bank:
                    # Flatten labels for memory bank
                    labels_sq = labels.squeeze(1)
                    
                    # Update Prototypes (FIXED: Logic inside memory bank skips 255)
                    self.memory_bank.update_prototypes(embedding.detach(), labels_sq)
                    
                    # Compute vMF Loss (FIXED: Logic inside proto_loss skips 255)
                    # Note: We do NOT overwrite labels with 999 here anymore!
                    from src.training.losses import compute_vmf_loss
                    loss_vmf = compute_vmf_loss(embedding, labels_sq, self.memory_bank, ignore_index=255)
                
                loss = loss_seg + 0.1 * loss_vmf

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"Seg": loss_seg.item(), "vMF": loss_vmf.item() if hasattr(loss_vmf, 'item') else 0})

            # --- SNAPSHOT every 100 steps ---
            if step % 100 == 0:
                # Quick inference for visualization
                with torch.no_grad():
                    # Get Energy Map for visualization
                    energy_map, _ = self.memory_bank.query_voxelwise_novelty(embedding)
                    preds = torch.argmax(logits, dim=1)
                    
                    self.vis.log_snapshot(inputs, labels, preds, energy_map, epoch, step)

        return epoch_loss / len(self.train_loader)

    def fit(self, max_epochs):
        for epoch in range(1, max_epochs + 1):
            loss = self.train_epoch(epoch)
            print(f"Epoch {epoch} Train Loss: {loss:.4f}")
            
            if epoch % self.config["training"]["eval_num"] == 0:
                self.validate(epoch)

    def validate(self, epoch):
        # ... (Your existing validation logic, but using self.dice_metric) ...
        # Add logic to save Best Model
        pass