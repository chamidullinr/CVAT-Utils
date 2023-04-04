import logging
import logging.config
import os

import yaml

from ..config import LOGGING_LEVEL

_module_dir = os.path.abspath(os.path.dirname(__file__))
LOGGER_CONFIG = os.path.join(_module_dir, "../config/logging.yaml")


def setup_logging():
    """Setup logging configuration from a file."""
    with open(LOGGER_CONFIG, "r") as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)

    logger = logging.getLogger("cvat_utils")
    logger.setLevel(LOGGING_LEVEL)
    for h in logger.handlers:
        h.setLevel(LOGGING_LEVEL)
