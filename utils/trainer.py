import torch
from utils.logger import get_logger
from utils.config import Config
from torchvision.ops import box_iou
from torchmetrics.detection import MeanAveragePrecision
from sklearn.metrics import  accuracy_score
import wandb
from datetime import datetime
import time

class Trainer:
    """Class to train Faster R-CNN model with PyTorch.
    
    Attributes:
        model (torch.nn.Module): The model to train.
        device (torch.device): The device to train the model on.
        optimizer (torch.optim.Optimizer): The optimizer to use.
        train_loader (torch.utils.data.DataLoader): The training data loader.
        val_loader (torch.utils.data.DataLoader): The validation data loader.
        test_loader (torch.utils.data.DataLoader): The test data loader.
        config (Config): The configuration object.
        logger (logging.Logger): The logger object.
        start_epoch (int): The starting epoch number.
        best_map (float): The best mAP score.
   
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        config: Config,
    ):
        """Initialize the Trainer object.
        
        Args:
            model (torch.nn.Module): The model to train.
            device (torch.device): The device to train the model on.
            optimizer (torch.optim.Optimizer): The optimizer to use.
            train_loader (torch.utils.data.DataLoader): The training data loader.
            val_loader (torch.utils.data.DataLoader): The validation data loader.
            test_loader (torch.utils.data.DataLoader): The test data loader.
            config (Config): The configuration object
        """
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.logger = get_logger(
            name="trainer", verbosity=config["trainer"]["verbosity"]
        )
        self.start_epoch = 0
        self.best_map = 0.0

        # Initialize wandb
        wandb.init(project="counting-trees", name=f"Model:{datetime.now().strftime('%d-%B-%Y-%I%p')}", config=config["wandb"])
        wandb.watch(self.model)

    def train(self):
        """Train the model."""
        for epoch in range(self.start_epoch, self.config["trainer"]["epochs"]):
            self.model.train()
            train_loss = 0.0
            start_time = time.time()
            for images, targets in self.train_loader:
                images = [image.to(self.device) for image in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                self.optimizer.zero_grad()
                loss_dict = self.model(images, targets)

                # Get all losses
                loss_box_reg = loss_dict["loss_box_reg"]
                loss_classifier = loss_dict["loss_classifier"]
                loss_objectness = loss_dict["loss_objectness"]
                loss_rpn_box_reg = loss_dict["loss_rpn_box_reg"]

                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                self.optimizer.step()
                
                train_loss += losses.item()
            
            self.lr_scheduler.step()
            end_time = time.time()

            epoch_time = end_time - start_time
            
            train_loss /= len(self.train_loader)

            metrics = {
                "Train/Loss": train_loss,
                "Train/loss_box_reg": loss_box_reg,
                "Train/loss_classifier": loss_classifier,
                "Train/loss_objectness": loss_objectness,
                "Train/loss_rpn_box_reg": loss_rpn_box_reg,
            }
            wandb.log({metrics})
            
            val_loss, val_map = self.validate(epoch)
            
            if val_map > self.best_map:
                self.best_map = val_map
                self.save_model("best_model")

            self.logger.info(f"Epoch: {epoch} Train Loss: {train_loss:.4f} Validation Loss: {val_loss:.4f} Validation mAP: {val_map:.4f} Time Elapsed: {int(epoch_time // 3600):02d}:{int((epoch_time % 3600) // 60):02d}:{int(epoch_time % 60):02d}")

        self.logger.info(f"Training finished. Best mAP: {self.best_map:.4f}")   

    def validate(self, epoch: int):
            """Validate the model."""
            self.model.eval()
            val_loss = 0.0
            all_predictions = []
            all_targets = []
            
            for images, targets in self.val_loader:
                images = [image.to(self.device) for image in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                loss_dict = self.model(images, targets)

                # Get all losses
                loss_box_reg = loss_dict["loss_box_reg"]
                loss_classifier = loss_dict["loss_classifier"]
                loss_objectness = loss_dict["loss_objectness"]
                loss_rpn_box_reg = loss_dict["loss_rpn_box_reg"]

                
                losses = sum(loss for loss in loss_dict.values())   # Total loss
                val_loss += losses.item()

                # Only get predictions
                predictions = self.model(images)
                
                preds = [
                    {"boxes": out["boxes"], "scores": out["scores"], "labels": out["labels"]} for out in predictions
                ]
                targs = [
                    {"boxes": tgt["boxes"], "labels": tgt["labels"]} for tgt in targets
                ]

                all_predictions.extend(preds)
                all_targets.extend(targs)

            val_loss /= len(self.val_loader)

            # Calculate metrics
            map_score = self.calculate_map(all_predictions, all_targets, "Validation")
            
            metrics = {
                "Validation/Loss": val_loss,
                "Validation/loss_box_reg": loss_box_reg,
                "Validation/loss_classifier": loss_classifier,
                "Validation/loss_objectness": loss_objectness,
                "Validation/loss_rpn_box_reg": loss_rpn_box_reg,
            }
            
            wandb.log(metrics)

            return val_loss, map_score

    def test(self):
        """Test the model.
        
        Returns:
            List[Dict]: A list containing the model predictions.
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in self.test_loader:
                images = [image.to(self.device) for image in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                
                # Only get predictions
                predictions = self.model(images)
                
                preds = [
                    {"boxes": out["boxes"], "scores": out["scores"], "labels": out["labels"]} for out in predictions
                ]
                targs = [
                    {"boxes": tgt["boxes"], "labels": tgt["labels"]} for tgt in targets
                ]

                all_predictions.extend(preds)
                all_targets.extend(targs)
                
        # Calculate metrics
        map_score = self.calculate_map(all_predictions, all_targets, "Test")
                
        self.logger.info(f"Test mAP: {map_score:.4f}")
        
        return all_predictions, all_targets

   
    def calculate_map(self, predictions, targets, eval_type,  iou_threshold=0.5):
        """Calculate mAP."""

        # Calculate mAP 
        map_metric = MeanAveragePrecision(iou_type="bbox")
        map_metric.update(predictions, targets)
        map_results = map_metric.compute()

        for k, v in map_results.items():
            wandb.log({f"{eval_type}/mAP/{k}": v} )

        return map_results["map"]

    def save_model(self, metric_name: str):
        """Save the model."""
        model_path = self.config["trainer"]["save_dir"] + f"models/{metric_name}.pth"
        torch.save(self.model, model_path)
        
        # Save the model as an artifact
        artifact = wandb.Artifact("model", type="model")
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)
        self.logger.info(f"Model saved at {model_path}")
        
        return model_path

    def load_model(self, model_path: str):
        """Load the model."""
        model = torch.load(model_path)
        self.logger.info(f"Model loaded from {model_path}")
        return model

    def __del__(self):
        """Finish the wandb run when the Trainer object is deleted."""
        wandb.finish()

    