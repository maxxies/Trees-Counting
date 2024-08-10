import numpy as np
import sys
import torch
import datetime
from torch.utils.data import Subset, DataLoader
from model.model import Model
from utils.data_loader import TreeDataset
from utils.config import Config
from utils.trainer import Trainer
from utils.logger import get_logger
from utils.visualisations import plot_sample_images

def run(working_dir: str, epochs: int):
    """Main function to run the training.
    
    Args:
        working_dir (str): The working directory where the data is stored.
    """
    # Define configurations
    cfg = {
        "main": {
            "name": "Counting Trees",
            "seed": 22,
            "cuda": True,
            "verbosity": 0,
        },
        "data": {
            "train_images_path": f"{working_dir}/train/",
            "test_images_path": f"{working_dir}/test/",
            "train_labels_path": f"{working_dir}/train_labels.csv",
            "test_labels_path": f"{working_dir}/test_labels.csv",
            "test_size": 0.2,
            "num_classes": 3, # 3 classes: background, tree, and palm tree
            "batch_size": 8,
        },
        "optimizer": {
            "type": "SGD",
            "args": {
                "lr": 0.005,
            },
        },
        "lr_scheduler": {
            "type": "StepLR",
            "args": {
                "step_size": 10,
                "gamma": 0.1,
            },
        },
        "trainer": {
            "epochs": epochs,
            "save_dir": "saved/",
            "verbosity": 0,
        },
        "logger": {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'brief': {
                    'format': '%(message)s'
                },
                'precise': {
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    'datefmt': '%Y-%m-%d %H:%M:%S'
                }
            },
            'handlers': {
                'console': {
                    'class': 'logging.StreamHandler',
                    'level': 'DEBUG',
                    'formatter': 'brief',
                    'stream': 'ext://sys.stdout'
                },
                'file': {
                    'class': 'logging.handlers.RotatingFileHandler',
                    'level': 'INFO',
                    'formatter': 'precise',
                    'filename': 'info.log',
                    'maxBytes': 1e+6,
                    'backupCount': 5,
                    'encoding': 'utf8'
                }
            },
            'root': {
                'level': 'INFO',
                'handlers': ['console', 'file']
            }
        },
        "wandb": {
            "architecture":"FasterRCNN",
            "dataset": "Palm Tree Dataset",
            "epochs": epochs,
        }
    }

    cfg = Config(cfg)
        
    # Set seed
    seed = cfg.config["main"]["seed"]
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    logger = get_logger(name="program", verbosity=cfg.config["main"]["verbosity"])


    # Set device
    device = torch.device("cuda" if cfg.config["main"]["cuda"] and torch.cuda.is_available() else "cpu")

    # Setup data_loader instances
    dataset = TreeDataset(cfg.config["data"]["train_images_path"], cfg.config["data"]["train_labels_path"], TreeDataset.get_transform(train=True))
    test_dataset = TreeDataset(cfg.config["data"]["test_images_path"], cfg.config["data"]["test_labels_path"], TreeDataset.get_transform(train=False))
    test_size = int(len(dataset) * cfg.config["data"]["test_size"])
    indices = torch.randperm(len(dataset)).tolist()
    train_dataset, val_dataset = Subset(dataset, indices[:-test_size]), Subset(dataset, indices[-test_size:])

    train_loader = DataLoader(train_dataset, batch_size=cfg.config["data"]["batch_size"], shuffle=True,  collate_fn=TreeDataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=cfg.config["data"]["batch_size"], shuffle=False,  collate_fn=TreeDataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=cfg.config["data"]["batch_size"], shuffle=False,  collate_fn=TreeDataset.collate_fn)
   
    # Plot sample images
    logger.info("Plotting sample images")
    plot_sample_images(train_loader, num_images=6)

    # Build model architecture
    model = Model(cfg.config["data"]["num_classes"]).get_model()
    model = model.to(device)

    # Build optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = getattr(torch.optim, cfg.config["optimizer"]["type"])(trainable_params, **cfg.config["optimizer"]["args"])

    # Build learning rate scheduler
    lr_scheduler = getattr(torch.optim.lr_scheduler, cfg.config["lr_scheduler"]["type"])(optimizer, **cfg.config["lr_scheduler"]["args"])

    # Log all the details
    logger.info(f"Using device: {device}")
    logger.info(f"Model set up: {type(model).__name__}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    total_params_size = sum(sys.getsizeof(p.untyped_storage()) for p in model.parameters())

    if total_params_size < 1024:
        logger.info(f"Size of Parameters: {total_params_size} bytes")
    elif total_params_size < (1024 ** 2):
        logger.info(f"Size of Parameters: {total_params_size / 1024:.2f} KB")
    elif total_params_size < (1024 ** 3):
        logger.info(f"Size of Parameters: {total_params_size / (1024 ** 2):.2f} MB")
    else:
        logger.info(f"Size of Parameters: {total_params_size / (1024 ** 3):.2f} GB")

    # Initialize the trainer
    trainer = Trainer(
        model=model,
        device=device,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=cfg.config,
    )

    # Start training
    trainer.train()

    # Test the model
    predictions, targets = trainer.test()

    # Return the test dataset and predictions
    return test_dataset, predictions, targets
