# Author: Chang Liu
# Some codes are refered from https://github.com/facebookresearch/pycls


"""Logging module."""


import logging
import sys
from typing import Optional


# Show filename and line number in logs
_FORMAT = "[%(filename)s: %(lineno)3d]: %(message)s"


def setup_logging(log_path: Optional[str]) -> None:
    """Setup the root logger by adding a handler and a formatter.

    Args: 
        None
    
    Returns:
        None
    """
    # Clear the root logger to prevent any existing logging config
    # (e.g. set by another module) from messing with our setup
    logging.root.handlers = []
    logging_config = {
        "level": logging.INFO, 
        "format": _FORMAT
    }
    if log_path:
        logging_config.update({"filename": log_path})
    else:
        logging_config.update({"stream": sys.stdout})
    logging.basicConfig(**logging_config)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with a specified name. Normally, the
    name is `__name__` of the module.

    Args:
        name (str):
            The name of the logger.
    
    Returns:
        logger (Logger):
            A Logger instance.
    """
    return logging.getLogger(name)



if __name__ == "__main__":
    pass