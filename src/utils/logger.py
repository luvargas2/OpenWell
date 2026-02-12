import logging
import os
import sys
import wandb

class ExperimentLogger:
    """
    Unified logger for Text (Console/File) and WandB (Cloud).
    """
    def __init__(self, log_dir, config=None, project_name="OpenWell"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # 1. Setup WandB
        try:
            wandb.init(
                project=project_name,
                name=os.path.basename(log_dir), # Use experiment folder name as run name
                config=config,
                dir=log_dir,
                resume="allow"
            )
            self.use_wandb = True
        except Exception as e:
            print(f"[WARNING] WandB failed to initialize: {e}")
            self.use_wandb = False
        
        # 2. Setup Python Logging (Text)
        self.logger = logging.getLogger("OpenWell")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False 
        
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        
        # Handler A: File
        file_handler = logging.FileHandler(os.path.join(log_dir, "training_log.txt"))
        file_handler.setFormatter(fmt)
        self.logger.addHandler(file_handler)
        
        # Handler B: Console
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(fmt)
        self.logger.addHandler(stream_handler)
        
        self.info(f"Experiment initialized at: {log_dir}")
        if self.use_wandb:
            self.info(f"WandB Run: {wandb.run.name}")

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def log_scalar(self, tag, value, step):
        """Logs a scalar value (Loss, Dice, LR)."""
        if self.use_wandb:
            wandb.log({tag: value}, step=step)

    def log_metrics(self, metrics_dict, step, prefix=""):
        """Logs a dictionary of metrics."""
        if self.use_wandb:
            # Flatten dict with prefix if needed
            log_dict = {f"{prefix}/{k}" if prefix else k: v for k, v in metrics_dict.items()}
            wandb.log(log_dict, step=step)

    def log_image(self, tag, image_path, step):
        """Logs an image file to WandB."""
        if self.use_wandb and os.path.exists(image_path):
            wandb.log({tag: wandb.Image(image_path)}, step=step)

    def close(self):
        if self.use_wandb:
            wandb.finish()
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)