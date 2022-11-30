import argparse
import logging
import os
import sys
from datetime import datetime
from typing import List, Tuple

import numpy as np
from tqdm.auto import tqdm

from cvat_utils import api_requests
from cvat_utils.core import download_images, load_task_data
from cvat_utils.utils import ErrorMonitor, to_json

logger = logging.getLogger("scripts")

SUPPORTED_SHAPES = ("points", "polyline", "polygon")


def load_args(args: list = None) -> argparse.Namespace:
    """Load script arguments using `argparse` library."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-ids",
        help="Task IDs in CVAT to load the data from.",
        type=int,
        required=True,
        nargs="+",
    )
    parser.add_argument(
        "--output-path",
        help="Output directory.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--load-images",
        help="If selected the script will download images from CVAT.",
        action="store_true",
    )
    parser.add_argument(
        "--points",
        help="If selected the script will include point annotations in the export.",
        action="store_true",
    )
    parser.add_argument(
        "--polylines",
        help="If selected the script will include polyline annotations in the export.",
        action="store_true",
    )
    parser.add_argument(
        "--polygons",
        help="If selected the script will include polygon annotations in the export.",
        action="store_true",
    )
    parser.add_argument(
        "--bboxes",
        help="If selected the script will include bbox annotations from polygon in the export.",
        action="store_true",
    )
    args = parser.parse_args(args)
    return args


def polygon2bbox(points: list) -> list:
    """Convert list of polygon points into a bounding box in a COCO format [xmin, ymin, w, h]."""
    pts = np.array(points).reshape(-1, 2)
    xmin, xmax = pts[:, 0].min(), pts[:, 0].max()
    ymin, ymax = pts[:, 1].min(), pts[:, 1].max()
    bbox = [float(xmin), float(ymin), float(xmax - xmin), float(ymax - ymin)]
    return bbox


def process_annotation_record(
    annot: dict,
    task_images: List[dict],
    id2label: dict,
    id2attrib: dict,
    *,
    error_monitor: ErrorMonitor,
    process_points: bool = False,
    process_polylines: bool = False,
    process_polygons: bool = False,
    process_bboxes: bool = False,
    is_tag: bool = False,
) -> dict:
    """Process a single record with annotation from CVAT.

    Parameters
    ----------
    annot
        A dictionary with annotation data from CVAT.
    task_images
        List of images in the current CVAT task.
    id2label
        A dictionary that maps label ids into label names.
    id2attrib
        A dictionary that maps attribute ids into attribute names.
    error_monitor
        An instance of `ErrorMonitor` class for storing errors for later.
    process_points
        If true process and return shapes of type `points`.
    process_polylines
        If true process and return shapes of type `polyline`.
    process_polygons
        If true process and return shapes of type `polygon`.
    process_bboxes
        If true process and return shapes of type `polygon` represented as a bounding box.

    Returns
    -------
    A dictionary with processed annotation record.
    """
    if is_tag or annot["type"] in SUPPORTED_SHAPES:
        # map frame index into image id
        frame_id = annot["frame"]
        image_id = task_images[frame_id]["id"]

        # map label id into label name
        assert (
            annot["label_id"] in id2label
        ), f"Error in CVAT - unknown label id {annot['label_id']}"
        label = id2label[annot["label_id"]]

        # map attribute ids into attribute names
        attributes = annot.get("attributes", [])
        for attr in attributes:
            assert (
                attr["spec_id"] in id2attrib
            ), f"Error in CVAT - unknown attribute id {attr['spec_id']}"
        attributes = {id2attrib[x["spec_id"]]: x["value"] for x in attributes}

        # process output
        if is_tag:  # process tag
            out = {
                "image_id": image_id,
                "type": "tag",
                "label": label,
                "attributes": attributes or None,
            }
        else:  # process shape
            out = {
                "image_id": image_id,
                "type": annot["type"],
                "label": label,
                "attributes": attributes or None,
            }
            if annot["type"] == "polygon" and process_bboxes:
                # store polygon record as bbox in COCO format [xmin, ymin, width, height]
                out["bbox"] = polygon2bbox(annot["points"])

            if process_points or process_polylines or process_polygons:
                # store raw record
                out["points"] = annot["points"]
    else:
        error_monitor.log_error(f"Unknown annotation type: {annot['type']}")
        out = None

    return out


def get_task_metadata(
    task_id: int,
    *,
    error_monitor: ErrorMonitor,
    process_points: bool = False,
    process_polylines: bool = False,
    process_polygons: bool = False,
    process_bboxes: bool = False,
    process_tags: bool = False,
) -> Tuple[List[dict], List[dict]]:
    """Load image metadata and annotations for the given task ID.

    Parameters
    ----------
    task_id
        Task ID in CVAT to load the data from.
    error_monitor
        An instance of `ErrorMonitor` class for storing errors for later.
    process_points
        If true process and return shapes of type `points`.
    process_polylines
        If true process and return shapes of type `polyline`.
    process_polygons
        If true process and return shapes of type `polygon`.
    process_bboxes
        If true process and return shapes of type `polygon` represented as a bounding box.
    process_tags
        If true process and return tags.

    Returns
    -------
    task_images
        A list with image metadata.
    task_annotations
        A list with annotations.
    """
    # load task data from CVAT
    task, jobs, task_images = load_task_data(task_id)
    id2label = {x["id"]: x["name"] for x in task["labels"]}
    id2attrib = {
        attr["id"]: attr["name"]
        for x in task["labels"]
        if "attributes" in x
        for attr in x["attributes"]
    }

    # load individual jobs and get annotations
    task_annotations = []
    for i, job in enumerate(jobs):
        # load annotations for a specific job
        url = os.path.join(job["url"], "annotations")
        annotations = api_requests.get(url)

        # process annotation records (shapes)
        for annot in annotations["shapes"]:
            # add job id to the image records
            frame_id = annot["frame"]
            task_images[frame_id]["job_id"] = job["id"]

            # process one record and store potential errors
            out = process_annotation_record(
                annot,
                task_images,
                id2label,
                id2attrib,
                error_monitor=error_monitor,
                process_points=process_points,
                process_polylines=process_polylines,
                process_polygons=process_polygons,
                process_bboxes=process_bboxes,
            )
            if out is not None:
                task_annotations.append(out)

        # process annotation records (tags)
        if process_tags:
            for annot in annotations["tags"]:
                # add job id to the image records
                frame_id = annot["frame"]
                task_images[frame_id]["job_id"] = job["id"]

                # process one record and store potential errors
                out = process_annotation_record(
                    annot,
                    task_images,
                    id2label,
                    id2attrib,
                    error_monitor=error_monitor,
                    is_tag=True,
                )
                if out is not None:
                    task_annotations.append(out)

    return task_images, task_annotations


def download_data(
    *,
    task_ids: List[int],
    output_path: str,
    load_images: bool = False,
    points: bool = False,
    polylines: bool = False,
    polygons: bool = False,
    bboxes: bool = False,
    tags: bool = False,
):
    """Download image metadata and annotations from CVAT and optionally images.

    Parameters
    ----------
    task_ids
        Task IDs in CVAT to load the data from.
    output_path
        Output directory.
    load_images
        If true load images together with image and annotation metadata.
    points
        If true process and return shapes of type `points`.
    polylines
        If true process and return shapes of type `polyline`.
    polygons
        If true process and return shapes of type `polygon`.
    bboxes
        If true process and return shapes of type `polygon` represented as a bounding box.
    tags
        If true process and return tags.
    """
    if not isinstance(task_ids, (list, tuple)):
        task_ids = [task_ids]

    # check CVAT credentials
    try:
        api_requests.load_credentials()
    except ValueError as e:
        logger.error(e)
        sys.exit(1)

    # create output paths
    os.makedirs(output_path, exist_ok=True)
    metadata_file = os.path.join(output_path, "metadata.json")
    if os.path.isfile(metadata_file):
        logger.warning(f"Directory {output_path} already has metadata.json file. Exiting.")
        sys.exit(0)

    images_path = None
    if load_images:
        images_path = os.path.join(output_path, "images")
        os.makedirs(images_path, exist_ok=True)

    # check filter arguments (at least one of them should be True)
    if not points and not polylines and not polygons and not bboxes:
        logger.warning(
            "Non of the filter arguments (points, polylines, polygons, bboxes) were selected. "
            "The script will download all available shape types."
        )
        points, polylines, polygons, bboxes = True, True, True, True

    # load data from CVAT
    logger.info(f"Processing tasks: {task_ids}")
    images, annotations = [], []
    error_monitor = ErrorMonitor()
    for task_id in tqdm(task_ids):
        # load task metadata including list of images and annotations
        task_images, task_annotations = get_task_metadata(
            task_id,
            error_monitor=error_monitor,
            process_points=points,
            process_polylines=polylines,
            process_polygons=polygons,
            process_bboxes=bboxes,
            process_tags=tags,
        )

        # download images
        if load_images:
            files = download_images(task_id, images_path)

            # include image paths to the metadata
            # remove task-{id}/images prefix to get a file id
            fileid2filepath = {
                x.split(".")[0].replace(f"task-{task_id}/images/", ""): x for x in files
            }
            for x in task_images:
                if x["id"] not in fileid2filepath:
                    error_monitor.log_error(f"Some images in task {task_id} were not downloaded.")
                x["file_path"] = fileid2filepath.get(x["id"])

        images.extend(task_images)
        annotations.extend(task_annotations)

    # print errors
    error_monitor.print_errors()

    if load_images:
        logger.info(f"Images were saved to the directory: {images_path}")

    # create output dictionary and save to JSON
    logger.info(f"Writing annotations to the metadata file: {metadata_file}")
    now = datetime.now()
    metadata = {
        "info": {
            "task_ids": task_ids,
            "date_created": now.strftime("%d/%m/%Y"),
        },
        "images": images,
        "annotations": annotations,
    }
    to_json(metadata, metadata_file)


if __name__ == "__main__":
    # load script args
    args = load_args()

    download_data(
        task_ids=args.task_ids,
        output_path=args.output_path,
        load_images=args.load_images,
        points=args.points,
        polylines=args.polylines,
        polygons=args.polygons,
        bboxes=args.bboxes,
        tags=args.tags,
    )
