import json
import logging

logger = logging.getLogger("cvat_utils")


def is_image(filename: str) -> bool:
    """Check if the filename has image extension."""
    return filename.lower().endswith((".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif"))


def to_json(obj: dict, filepath: str):
    """Save a dictionary to a JSON file."""
    with open(filepath, "w") as f:
        json.dump(obj, f)


def read_json(filepath: str) -> dict:
    """Read a JSON file and return dictionary."""
    with open(filepath, "r") as f:
        return json.load(f)


class ErrorMonitor:
    """Object for tracking errors and counting number of their occurrences."""

    def __init__(self):
        self.errors = {}

    def log_error(self, msg: str):
        """Save an error."""
        if msg not in self.errors:
            self.errors[msg] = 0
        self.errors[msg] += 1

    def print_errors(self):
        """Log an output with errors."""
        if len(self.errors) > 0:
            logger.warning(
                f"Found the following errors during processing: {json.dumps(self.errors, indent=4)}"
            )
