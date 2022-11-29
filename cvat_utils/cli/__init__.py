import fire

from .download import download_data


def app() -> None:
    """Command line interface entry point used by the `cvat_utils` package."""
    fire.Fire({"download": download_data})
