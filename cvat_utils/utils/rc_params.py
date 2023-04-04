import logging
import os
import re

from dotenv import load_dotenv

from .. import config

logger = logging.getLogger("cvat_utils")


class RcParams(dict):
    def __init__(self):
        self.config_variables = [x for x in dir(config) if re.fullmatch(r"[A-Z_]*", x)]
        super().__init__({x: getattr(config, x) for x in self.config_variables})

        # update variables from global environment variables
        # get environment variables from .env
        is_success = load_dotenv(dotenv_path=os.path.join(os.getcwd(), ".env"))
        logger.debug(f"Loaded environment files from `.env` file: {is_success}")
        for x in self.config_variables:
            if x in os.environ:
                self[x] = os.environ[x]

    def __setitem__(self, key, value):
        if key not in self.config_variables:
            raise KeyError(
                f"Unknown key: '{key}'. Use one of the config keys: {self.config_variables}."
            )
        logger.info(f"Update configuration: {key}={value}.")
        setattr(config, key, value)
        super().__setitem__(key, value)


rc_params = RcParams()
