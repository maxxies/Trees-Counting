import logging
import logging.config
from pathlib import Path

LOG_LEVELS = {
    0: logging.DEBUG,
    1: logging.INFO,
    2: logging.WARNING,
    3: logging.ERROR,
}

def global_logger_setup(log_cfg: dict, log_dir: str | Path) -> None:
    """Setup global logging. All loggers will inherit this setup.

    Args:
        log_cfg (dict): The logging configuration dictionary.
        log_dir (str | Path): The directory to save the logs.
    """
    for _, handler in log_cfg["handlers"].items():
        if "filename" in handler:
            handler["filename"] = str(log_dir / handler["filename"])
    logging.config.dictConfig(log_cfg)


def get_logger(name: str, verbosity: int = 2) -> logging.Logger:
    """Instantiate logger with the specified name and verbosity level.

    Args:
        name (str): The name of the logger.
        verbosity (int, optional): The verbosity level of the logger. Defaults to 2.

    Returns:
        logging.Logger: The logger with the specified name and verbosity level.

    Raises:
        AssertionError: If the verbosity level is not within the valid range (0-3).
    """
    assert (
        verbosity in LOG_LEVELS
    ), f"Verbosity option {verbosity} is out of range (0-3)"
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVELS[verbosity])
    return logger
