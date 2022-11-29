import io
import logging
import os
import time
import zipfile
from typing import List, Tuple

from cvat_utils import api_requests
from cvat_utils.utils import is_image

logger = logging.getLogger("cvat_utils")


def load_task_data(task_id: int) -> Tuple[dict, List[dict], List[dict]]:
    """Load task metadata from CVAT."""
    # load annotation data from CVAT
    task_url = f"https://cvat.piva-ai.com/api/v1/tasks/{task_id}"
    task = api_requests.get(task_url)
    meta = api_requests.get(task_url + "/data/meta")

    # get list of jobs in the task
    jobs = [
        {
            "id": job["id"],
            "url": job["url"].replace("http://", "https://"),
            "status": job["status"],
            "start_frame": segment["start_frame"],
            "stop_frame": segment["stop_frame"],
        }
        for segment in task["segments"]
        for job in segment["jobs"]
    ]

    # get list of frames
    frames = [
        {
            "id": x["name"].split(".")[0],
            "file_name": x["name"].split("/")[-1],
            "width": x["width"],
            "height": x["height"],
            "task_id": task_id,
            "task_name": task["name"],
        }
        for x in meta["frames"]
    ]

    return task, jobs, frames


def download_images(task_id: int, output_path: str) -> list:
    """Download images from CVAT and save them to a local directory.

    Parameters
    ----------
    task_id
        CVAT Task ID to load images from.
    output_path
        A local directory where images will be saved.

    Returns
    -------
    A list of downloaded images.
    """
    url = f"https://cvat.piva-ai.com/api/v1/tasks/{task_id}/dataset"

    # create request and wait till 201 (created) status code
    while True:
        resp = api_requests.get(url, params={"format": "CVAT for images 1.1"}, load_content=False)
        if resp.status_code == 201:
            break
        if resp.status_code == 500:
            logger.error(f"Error: receiver response 500 with content: {resp.content}")
        time.sleep(5)

    # load images
    resp = api_requests.get(
        url,
        params={
            "format": "CVAT for images 1.1",
            "action": "download",
        },
        load_content=False,
    )

    # extract zip file
    output_path = os.path.join(output_path, f"task-{task_id}")
    os.makedirs(output_path, exist_ok=False)
    buffer = io.BytesIO(resp.content)
    with zipfile.ZipFile(buffer) as zip_ref:
        files = [x for x in zip_ref.namelist() if is_image(x)]
        zip_ref.extractall(output_path, members=files)
    files = [os.path.join(f"task-{task_id}", x) for x in files]
    # logger.info(f"Downloaded and extracted {len(files)} files to '{output_path}'.")

    return files
