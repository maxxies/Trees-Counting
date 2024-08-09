from pathlib import Path
from datetime import datetime
from utils.logger import global_logger_setup

class Config:
    """ Configuration handler."""

    def __init__(self, config: dict):
        """Initialize configuration from a dictionary.

        Args:
            config (dict): The configuration dictionary.
        """
        self._config = config

        # Set experiment name and run id
        exp_name = self.config["main"]["name"]
        run_id = datetime.now().strftime(r"%Y%m%d_%H%M%S")

        # Set and create directory for saving log and model
        save_dir = Path(self.config["trainer"]["save_dir"])
        self._save_dir: Path = save_dir / "models" / exp_name / run_id
        self.log_dir: Path = save_dir / "logs" / exp_name / run_id

        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        global_logger_setup(self.config["logger"], self.log_dir)


    @property
    def config(self):
        return self._config
    
    @property
    def save_dir(self):
        return self._save_dir
    

