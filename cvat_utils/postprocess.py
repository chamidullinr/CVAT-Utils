import logging
import os

from sklearn.model_selection import train_test_split

from .utils import to_json

logger = logging.getLogger("cvat_utils")


def save_metadata(
    metadata: dict,
    *,
    output_path: str = None,
    file_name: str = "metadata.json",
    train_file_name: str = "train.json",
    test_file_name: str = "test.json",
    split_dataset: bool = False,
    test_size: float = 0.2,
):
    """Save metadata dictionary to JSON - either to a single file or train and test files.

    Parameters
    ----------
    metadata
        A dictionary with dataset metadata. It should contain `images` and `annotations` fields.
    output_path
        A directory to save the metadata file(s) to.
    file_name
        A metadata file name to use for a single file.
    train_file_name
        A metadata file name to use for a training set.
    test_file_name
        A metadata file name to use for a test set.
    split_dataset
        If true split the dataset into training and test sets.
    test_size
        If split_dataset=true specifies size in % of the test set.
    """
    assert "images" in metadata
    assert "annotations" in metadata
    images = metadata["images"]
    annotations = metadata["annotations"]

    if split_dataset:
        # create train test split
        logger.info("Splitting metadata into training and test splits.")
        assert isinstance(test_size, float) and 0 < test_size < 1
        train_images, test_images = train_test_split(images, test_size=test_size)
        train_image_ids = set([x["id"] for x in train_images])
        train_metadata = {
            "info": metadata["info"],
            "images": train_images,
            "annotations": [x for x in annotations if x["image_id"] in train_image_ids],
        }
        test_metadata = {
            "info": metadata["info"],
            "images": test_images,
            "annotations": [x for x in annotations if x["image_id"] not in train_image_ids],
        }

        # write to the output files
        if output_path is not None:
            train_file_name = os.path.join(output_path, train_file_name)
            test_file_name = os.path.join(output_path, test_file_name)
        logger.info(
            f"Saving training metadata file '{train_file_name}' "
            f"with {len(train_images)} images "
            f"and {len(train_metadata['annotations'])} annotations."
        )
        logger.info(
            f"Saving test metadata file '{test_file_name}' "
            f"with {len(test_images)} images "
            f"and {len(test_metadata['annotations'])} annotations."
        )
        to_json(train_metadata, train_file_name)
        to_json(test_metadata, test_file_name)
    else:
        # write to the output file
        file_name = os.path.join(output_path, file_name)
        logger.info(
            f"Saving metadata file '{file_name}' "
            f"with {len(images)} images and {len(annotations)} annotations."
        )
        to_json(metadata, file_name)
