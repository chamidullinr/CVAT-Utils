import logging
import logging.config
import os

import yaml

from .. import config

_module_dir = os.path.abspath(os.path.dirname(__file__))
LOGGER_CONFIG = os.path.join(_module_dir, "../config/logging.yaml")


def setup_logging():
    """Setup logging configuration from a file."""
    if not logging.getLogger("cvat_utils").hasHandlers():
        with open(LOGGER_CONFIG, "r") as f:
            logger_config = yaml.safe_load(f.read())
            logging.config.dictConfig(logger_config)

        logger = logging.getLogger("cvat_utils")
        logger.setLevel(config.LOGGING_LEVEL)
        for h in logger.handlers:
            h.setLevel(config.LOGGING_LEVEL)
