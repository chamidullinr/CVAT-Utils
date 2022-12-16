import logging

from cvat_utils.log import setup_logging

from . import api_requests, core, utils
from .core import download_images, load_task_data
from .version import __version__

__all__ = ["api_requests", "core", "utils", "download_images", "load_task_data", "__version__"]

setup_logging()
logger = logging.getLogger("cvat_utils")
