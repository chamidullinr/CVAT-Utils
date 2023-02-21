import argparse
import logging
import os
import shutil
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Union

import numpy as np
from tqdm.auto import tqdm

from cvat_utils import download_images, load_annotations, load_credentials, load_task_data
from cvat_utils.models import Frame, FullShape, FullTag
from cvat_utils.utils import ErrorMonitor, to_json

logger = logging.getLogger("script")

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
    parser.add_argument(
        "--all-jobs",
        help="If selected the script will include all jobs regardless the status. "
        "By default the script includes only jobs with status=completed",
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
    annot: Union[FullShape, FullTag],
    task_images: Dict[int, Frame],
    id2label: dict,
    id2attrib: dict,
    *,
    error_monitor: ErrorMonitor,
    process_points: bool = False,
    process_polylines: bool = False,
    process_polygons: bool = False,
    process_bboxes: bool = False,
) -> dict:
    """Process a single record with annotation from CVAT.

    Parameters
    ----------
    annot
        A dictionary with annotation data from CVAT.
    task_images
        Dictionary of images in the current CVAT task.
        Keys correspond to frame ids in the task and values contain dictionary with image metadata.
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
    is_tag = isinstance(annot, FullTag)
    if is_tag or annot.type in SUPPORTED_SHAPES:
        # map frame index into image id
        image_id = task_images[annot.frame].id

        # map label id into label name
        assert annot.label_id in id2label, f"Error in CVAT - unknown label id {annot.label_id}"
        label = id2label[annot.label_id]

        # map attribute ids into attribute names
        for attr in annot.attributes:
            assert (
                attr["spec_id"] in id2attrib
            ), f"Error in CVAT - unknown attribute id {attr['spec_id']}"
        attributes = {id2attrib[x["spec_id"]]: x["value"] for x in annot.attributes}

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
                "type": annot.type,
                "label": label,
                "attributes": attributes or None,
            }
            if annot.type == "polygon" and process_bboxes:
                # store polygon record as bbox in COCO format [xmin, ymin, width, height]
                out["bbox"] = polygon2bbox(annot.points)

            if (
                (annot.type == "points" and process_points)
                or (annot.type == "polyline" and process_polylines)
                or (annot.type == "polygon" and process_polygons)
            ):
                # store raw record
                out["points"] = annot.points
    else:
        error_monitor.log_error(f"Unknown annotation type: {annot.type}")
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
    all_jobs: bool = False,
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
    all_jobs
        If true include all jobs regardless the status.
        By default, include only jobs with status=completed.

    Returns
    -------
    task_images
        A list with image metadata.
    task_annotations
        A list with annotations.
    """
    # load task data from CVAT
    task, jobs, task_images = load_task_data(task_id)
    id2label = {x.id: x.name for x in task.labels}
    id2attrib = {attr["id"]: attr["name"] for x in task.labels for attr in x.attributes}

    # drop task images from jobs that are not completed yet
    if not all_jobs:
        task_images = {k: v for k, v in task_images.items() if v.status == "completed"}

    # load annotation data from individual jobs
    task_annotations = []
    for i, job in enumerate(jobs):
        # skip if the job is not completed yet
        if not all_jobs and job.status != "completed":
            continue

        # load annotations for a specific job
        annotations = load_annotations(job)

        # process annotation records (shapes) and store potential errors
        for annot in annotations.shapes:
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

        # process annotation records (tags) and store potential errors
        if process_tags:
            for annot in annotations.tags:
                out = process_annotation_record(
                    annot,
                    task_images,
                    id2label,
                    id2attrib,
                    error_monitor=error_monitor,
                )
                if out is not None:
                    task_annotations.append(out)

    # convert task images to list
    task_images = list(task_images.values())

    # convert to dict
    task_images = [x.dict() for x in task_images]

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
    all_jobs: bool = False,
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
    all_jobs
        If true include all jobs regardless the status.
        By default, include only jobs with status=completed.
    """
    if not isinstance(task_ids, (list, tuple)):
        task_ids = [task_ids]

    # check CVAT credentials
    try:
        load_credentials()
    except ValueError as e:
        logger.error(e)
        sys.exit(1)

    # create output paths
    os.makedirs(output_path, exist_ok=True)
    metadata_file = os.path.join(output_path, "metadata.json")
    if os.path.isfile(metadata_file):
        logger.warning(f"Directory {output_path} already has metadata.json file. Exiting.")
        sys.exit(0)

    images_path, images_tmp_path = None, None
    if load_images:
        images_path = os.path.join(output_path, "images")
        images_tmp_path = os.path.join(output_path, "images_tmp")
        os.makedirs(images_path, exist_ok=True)
        os.makedirs(images_tmp_path, exist_ok=False)

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
            all_jobs=all_jobs,
        )

        # download images
        if load_images:
            # download images to the temporary directory
            files = download_images(task_id, images_tmp_path)

            # align downloaded images with metadata
            # remove task-{id}/images prefix to get a file id
            imageid2filepath = {
                x.split(".")[0].replace(f"task-{task_id}/images/", ""): x for x in files
            }
            for image_data in task_images:
                image_id = image_data["id"]
                if image_id not in imageid2filepath:
                    error_monitor.log_error(f"Some images in task {task_id} were not downloaded.")
                    image_data["file_path"] = None
                else:
                    file_path = imageid2filepath[image_id]
                    image_ext = image_data["file_name"].split(".")[-1]
                    new_file_path = f"task-{task_id}/{image_id}.{image_ext}"

                    # move to the image directory
                    trg = os.path.join(images_path, new_file_path)
                    os.makedirs(os.path.dirname(trg), exist_ok=True)
                    os.rename(
                        os.path.join(images_tmp_path, file_path),
                        trg,
                    )

                    # include image paths to the metadata
                    image_data["file_path"] = new_file_path

            # remove temporary directory with unused images
            shutil.rmtree(images_tmp_path)

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

    if error_monitor.has_errors():
        sys.exit(1)


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
        all_jobs=args.all_jobs,
    )
