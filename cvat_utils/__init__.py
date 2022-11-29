import logging

from dotenv import load_dotenv

from cvat_utils.log import setup_logging

from . import api_requests, core, utils
from .version import __version__

__all__ = ["api_requests", "core", "utils", "__version__"]

load_dotenv()  # get environment variables from .env

setup_logging()
logger = logging.getLogger("cvat_utils")
