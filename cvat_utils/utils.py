import json
import logging

logger = logging.getLogger("cvat_utils")


def is_image(filename):
    return filename.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"))


def to_json(obj: dict, filepath: str):
    with open(filepath, "w") as f:
        json.dump(obj, f)


def read_json(filepath: str) -> dict:
    with open(filepath, "r") as f:
        return json.load(f)


class ErrorMonitor:
    def __init__(self):
        self.errors = {}

    def log_error(self, msg: str):
        if msg not in self.errors:
            self.errors[msg] = 0
        self.errors[msg] += 1

    def print_errors(self):
        logger.warning(
            f"Found the following errors during processing: {json.dumps(self.errors, indent=4)}"
        )
