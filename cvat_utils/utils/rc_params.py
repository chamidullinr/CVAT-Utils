import re

from .. import config


class RcParams(dict):
    def __init__(self):
        self.config_variables = [x for x in dir(config) if re.fullmatch(r"[A-Z_]*", x)]
        super().__init__({x: getattr(config, x) for x in self.config_variables})

    def __setitem__(self, key, value):
        if key not in self.config_variables:
            raise KeyError(
                f"Unknown key: '{key}'. Use one of the config keys: {self.config_variables}."
            )
        setattr(config, key, value)
        super().__setitem__(key, value)


rc_params = RcParams()
