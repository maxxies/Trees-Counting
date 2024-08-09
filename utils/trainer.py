import torch
from utils.logger import get_logger
from utils.config import Config

class Trainer:
    """Class to train Faster R-CNN model with PyTorch.

    Attributes:
        model (Model): The Faster R-CNN model.
        device (torch.device): The device to run the model on.
        optimizer (torch.optim.Optimizer): The optimizer for training.
        criterion (torch.nn.Module): The loss function.
        train_loader (torch.utils.data.DataLoader): The training data loader.
        val_loader (torch.utils.data.DataLoader): The validation data loader.
        test_loader (torch.utils.data.DataLoader): The test data loader.
        config (Config): The configuration object.
        logger (logging.Logger): The logger object.
        tensorboard (TensorboardWriter): The Tensorboard writer object.
        start_epoch (int): The starting epoch for training.
        best_loss (float): The best loss achieved during training.
        best_metric (float): The best metric achieved during training.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        config: Config,
    ):
        """Initialize the Trainer object.

        Args:
            model (Model): The Faster R-CNN model.
            device (torch.device): The device to run the model on.
            optimizer (torch.optim.Optimizer): The optimizer for training.
            criterion (torch.nn.Module): The loss function.
            train_loader (torch.utils.data.DataLoader): The training data loader.
            val_loader (torch.utils.data.DataLoader): The validation data loader.
            test_loader (torch.utils.data.DataLoader): The test data loader.
            config (Config): The configuration object.
        """
        self.model = model
        self.device = device

        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.logger = get_logger(
            name="trainer", verbosity=config["trainer"]["verbosity"]
        )

        self.start_epoch = 0
        self.best_loss = float("inf")
        self.best_metric = 0.0

    def train(self):
        """Train the model."""
        for epoch in range(self.start_epoch, self.config["trainer"]["epochs"]):
            self.model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            for images, targets in self.train_loader:
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                self.optimizer.zero_grad()
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                losses.backward()
                self.optimizer.step()
                train_loss += losses.item()
            train_loss /= len(self.train_loader)
            self.tensorboard.add_scalar("train_loss", train_loss, epoch)
            self.logger.info(f"Epoch {epoch}: Train Loss: {train_loss}")
            self.validate(epoch)

    def validate(self, epoch: int):
        """Validate the model."""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, targets in self.val_loader:
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                val_loss += losses.item()

        val_loss /= len(self.val_loader)
        self.tensorboard.add_scalar("val_loss", val_loss, epoch)
        self.logger.info(f"Epoch {epoch}: Validation Loss: {val_loss}")
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.save_model(epoch, val_loss, "best_loss")
        self.test()

    def test(self):
        """Test the model."""
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, targets in self.test_loader:
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                test_loss += losses.item()
              
        test_loss /= len(self.test_loader)
        self.tensorboard.add_scalar("test_loss", test_loss)
        self.logger.info(f"Test Loss: {test_loss}")

    def save_model(self, epoch: int, metric: float, metric_name: str):
        """Save the model."""
        model_dir = self.config.save_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{metric_name}.pth"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metric": metric,
            },
            model_path,
        )
        self.logger.info(f"Model saved at {model_path}")      
        return model_path

    def load_model(self, model_path: str):
        """Load the model."""
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_metric = checkpoint["metric"]
        self.logger.info(f"Model loaded from {model_path}")

   