import logging

from cvat_utils.log import setup_logging

from . import api_requests, core, utils
from .api_requests import load_credentials
from .core import (
    download_images,
    image_path_to_image_id,
    load_annotations,
    load_project_data,
    load_task_data,
)
from .version import __version__

__all__ = [
    "api_requests",
    "core",
    "utils",
    "load_credentials",
    "download_images",
    "load_project_data",
    "load_task_data",
    "load_annotations",
    "image_path_to_image_id",
    "__version__",
]

setup_logging()
logger = logging.getLogger("cvat_utils")
