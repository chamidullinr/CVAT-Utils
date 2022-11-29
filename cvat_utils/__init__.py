import logging

from cvat_utils.log import setup_logging
from .version import __version__

__all__ = ["__version__"]

setup_logging()
logger = logging.getLogger("cvat_utils")
