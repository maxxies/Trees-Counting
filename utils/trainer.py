import torch
from utils.logger import get_logger
from utils.config import Config
from torchvision.ops import box_iou
from torchmetrics.detection import MeanAveragePrecision
from sklearn.metrics import  accuracy_score
import wandb
from datetime import datetime

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
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.logger = get_logger(
            name="trainer", verbosity=config.trainer.verbosity
        )
        self.start_epoch = 0
        self.best_map = 0.0

        # Initialize wandb
        wandb.init(project="counting-trees", name=f"Model:{datetime.now().strftime('%d-%B-%Y-%I%p')}", config=config["wandb"])
        wandb.watch(self.model)

    def train(self):
        """Train the model."""
        for epoch in range(self.start_epoch, self.config.trainer.epochs):
            self.model.train()
            train_loss = 0.0
            for images, targets in self.train_loader:
                images = [image.to(self.device) for image in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                self.optimizer.zero_grad()
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                self.optimizer.step()
                
                train_loss += losses.item()
            
            train_loss /= len(self.train_loader)
            wandb.log({"Train/Loss": train_loss, "epoch": epoch})
            self.logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}")
            
            val_map = self.validate(epoch)
            
            if val_map > self.best_map:
                self.best_map = val_map
                self.save_model(epoch, val_map, "best_map")

        self.logger.info(f"Training finished. Best mAP: {self.best_map:.4f}")   

    def validate(self, epoch: int):
        """Validate the model."""
        self.model.eval()
        val_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = [image.to(self.device) for image in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()
                
                predictions = self.model(images)
                all_predictions.extend(predictions)
                all_targets.extend(targets)

        val_loss /= len(self.val_loader)
        wandb.log({"Validation/Loss": val_loss, "epoch": epoch})
        self.logger.info(f"Epoch {epoch}: Validation Loss: {val_loss:.4f}")

        # Calculate metrics
        map_score, accuracy, precision, recall, f1 = self.calculate_metrics(all_predictions, all_targets)
        
        metrics = {
            "Validation/mAP": map_score,
            "Validation/Accuracy": accuracy,
            "Validation/Precision": precision,
            "Validation/Recall": recall,
            "Validation/F1": f1,
            "epoch": epoch
        }
        wandb.log(metrics)
        
        self.logger.info(f"Epoch {epoch}: Validation mAP: {map_score:.4f}, Accuracy: {accuracy:.4f}, "
                         f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        return map_score

    def test(self):
        """Test the model.
        
        Returns:
            List[Dict]: A list containing the model predictions.
        """
        self.model.eval()
        test_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in self.test_loader:
                images = [image.to(self.device) for image in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                test_loss += losses.item()
                
                predictions = self.model(images)
                all_predictions.extend(predictions)
                all_targets.extend(targets)

        test_loss /= len(self.test_loader)
        wandb.log({"Test/Loss": test_loss})
        self.logger.info(f"Test Loss: {test_loss:.4f}")

        # Calculate metrics
        map_score, accuracy, precision, recall, f1 = self.calculate_metrics(all_predictions, all_targets)
        
        metrics = {
            "Test/mAP": map_score,
            "Test/Accuracy": accuracy,
            "Test/Precision": precision,
            "Test/Recall": recall,
            "Test/F1": f1
        }
        wandb.log(metrics)
        
        self.logger.info(f"Test mAP: {map_score:.4f}, Accuracy: {accuracy:.4f}, "
                         f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        return all_predictions

    def calculate_metrics(self, predictions, targets, iou_threshold=0.5):
        """Calculate mAP, accuracy, precision, recall, and F1 score."""
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        all_pred_labels = []
        all_true_labels = []
        
        for pred, target in zip(predictions, targets):
            pred_boxes = pred['boxes']
            pred_labels = pred['labels']
            pred_scores = pred['scores']
            
            target_boxes = target['boxes']
            target_labels = target['labels']
            
            iou = box_iou(pred_boxes, target_boxes)
            max_iou, max_idx = iou.max(dim=1)
            
            for i, (iou_val, pred_label, pred_score) in enumerate(zip(max_iou, pred_labels, pred_scores)):
                all_pred_labels.append(pred_label.item())
                true_label = target_labels[max_idx[i]].item()
                all_true_labels.append(true_label)
                
                if iou_val >= iou_threshold and pred_label == true_label:
                    true_positives += 1
                else:
                    false_positives += 1
            
            false_negatives += len(target_boxes) - true_positives

        precision = true_positives / (true_positives + false_positives + 1e-6)
        recall = true_positives / (true_positives + false_negatives + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        
        # Calculate accuracy
        accuracy = accuracy_score(all_true_labels, all_pred_labels)
        
        # Calculate mAP 
        map_score = MeanAveragePrecision(iou_type="bbox")
        map_score.update(predictions, target)
        map = map_score.compute()

    
        return map, accuracy, precision, recall, f1

    def save_model(self, metric_name: str):
        """Save the model."""
        model_dir = self.config.save_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{metric_name}.pth"
        torch.save(self.model, model_path)
        self.logger.info(f"Model saved at {model_path}")
        
        wandb.save(str(model_path))
        return model_path

    def load_model(self, model_path: str):
        """Load the model."""
        model = torch.load(model_path)
        self.logger.info(f"Model loaded from {model_path}")
        return model

    def __del__(self):
        """Finish the wandb run when the Trainer object is deleted."""
        wandb.finish()