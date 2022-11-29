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
        "--images",
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


def polygon2bbox(points: list):
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
) -> dict:
    if annot["type"] in SUPPORTED_SHAPES:
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
    points: bool = False,
    polylines: bool = False,
    polygons: bool = False,
    bboxes: bool = False,
) -> Tuple[List[dict], List[dict]]:
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

        # process annotation records
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
                process_points=points,
                process_polylines=polylines,
                process_polygons=polygons,
                process_bboxes=bboxes,
            )
            if out is not None:
                task_annotations.append(out)

    return task_images, task_annotations


def download_data(
    *,
    task_ids: List[int],
    output_path: str,
    images: bool = False,
    points: bool = False,
    polylines: bool = False,
    polygons: bool = False,
    bboxes: bool = False,
):
    # create output paths
    os.makedirs(output_path, exist_ok=True)
    metadata_file = os.path.join(output_path, "metadata.json")
    if os.path.isfile(metadata_file):
        logger.warning(f"Directory {output_path} already has metadata.json file. Exiting.")
        sys.exit(0)

    images_path = None
    if images:
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
            points=points,
            polylines=polylines,
            polygons=polygons,
            bboxes=bboxes,
        )

        # download images
        if images:
            files = download_images(task_id, images_path)

            # include image paths to the metadata
            fileid2filepath = {"/".join(x.split(".")[0].split("/")[1:]): x for x in files}
            for x in task_images:
                if x["id"] not in fileid2filepath:
                    error_monitor.log_error(f"Some images in task {task_id} were not downloaded.")
                x["file_path"] = fileid2filepath.get(x["id"])

        images.extend(task_images)
        annotations.extend(task_annotations)

    # print errors
    error_monitor.print_errors()

    if images:
        logger.info(f"Images were saved to the directory: {images_path}")

    # create output dictionary and save to JSON
    logger.info(f"Writing annotations to the metadata file: {metadata_file}")
    now = datetime.now()
    metadata = {
        "info": {
            "task_ids": task_ids,
            "date_created": now.strftime("%Y/%m/%d"),
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
        images=args.images,
        points=args.points,
        polylines=args.polylines,
        polygons=args.polygons,
        bboxes=args.bboxes,
    )
