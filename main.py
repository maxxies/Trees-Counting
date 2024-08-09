import numpy as np
import sys
import torch
from torch.utils.data import Subset, DataLoader
from model.model import Model
from utils.data_loader import TreeDataset
from utils.config import Config
from utils.trainer import Trainer
from utils.logger import get_logger

def main(woriking_dir: str):
    """Main function to run the training.
    
    Args:
        woriking_dir (str): The working directory where the data is stored.
    """
    # Define configurations
    cfg = {
        "main": {
            "name": "Counting Trees",
            "cuda": True,
            "verbosity": 0,
        },
        "data": {
            "train_images_path": f"{woriking_dir}/train/",
            "test_images_path": f"{woriking_dir}/test/",
            "train_labels_path": f"{woriking_dir}/train_labels.csv",
            "test_labels_path": f"{woriking_dir}/test_labels.csv",
            "test_size": 0.2,
            "num_classes": 3, # 3 classes: background, tree, and palm tree
            "batch_size": 16,
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
            "epochs": 50,
            "log_step": 1700,
            "save_dir": "saved/",
            "save_period": 50,
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
            "log_config": True,
            "sync_tensorboard": True,
            "reinit": True,
        }
    }

        
    # Set seed
    seed = cfg.main.seed
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    # Set device
    device = torch.device("cuda" if cfg.main.cuda and torch.cuda.is_available() else "cpu")

    # Setup data_loader instances
    dataset = TreeDataset(cfg.data.train_images_path, cfg.data.train_labels_path, TreeDataset.get_transform(train=True))
    test_dataset = TreeDataset(cfg.data.test_images_path, cfg.data.test_labels_path, TreeDataset.get_transform(train=False))
    test_size = int(len(dataset) * cfg.data.test_size)
    indices = torch.randperm(len(dataset)).tolist()
    train_dataset, val_dataset = Subset(dataset, indices[:-test_size]), Subset(dataset, indices[-test_size:])

    train_loader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, shuffle=True, num_workers=4, collate_fn=TreeDataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=4, collate_fn=TreeDataset.collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=cfg.data.batch_size, shuffle=False, num_workers=4, collate_fn=TreeDataset.collate_fn)

    # Build model architecture
    model = Model(cfg.data.num_classes)
    model = model.to(device)

    # Build optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = getattr(torch.optim, cfg.optimizer.type)(trainable_params, **cfg.optimizer.args)

    # Build learning rate scheduler
    lr_scheduler = getattr(torch.optim.lr_scheduler, cfg.lr_scheduler.type)(optimizer, **cfg.lr_scheduler.args)

    # Log all the details
    logger = get_logger(name="program", verbosity=cfg.main.verbosity)
    logger.info(f"Using device: {device}")
    logger.info(f"Model set up: {type(model).__name__}")
    logger.info(f"Trainable parameters: {sum(p.numel() for p in trainable_params)}")

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
    cfg = Config(cfg)

    trainer = Trainer(
        model=model,
        device=device,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=cfg,
    )

    trainer.train()