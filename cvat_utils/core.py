import io
import logging
import os
import time
import warnings
import zipfile
from typing import Dict, List, Tuple, Union

from cvat_utils import api_requests
from cvat_utils.models import (
    Frame,
    FullAnnotations,
    FullProject,
    FullTask,
    FullTaskMetadata,
    Job,
    Task,
)
from cvat_utils.utils import is_image

logger = logging.getLogger("cvat_utils")


def _load_project(project_id: int) -> FullProject:
    project_url = f"https://cvat.piva-ai.com/api/v1/projects/{project_id}"
    project_dict = api_requests.get(project_url)
    project = FullProject(**project_dict)
    if project.dict() != project_dict:
        warnings.warn(
            "Project model in the library doesn't equal to the model returned by CVAT API."
        )
    return project


def _load_task(task_id: int) -> FullTask:
    task_url = f"https://cvat.piva-ai.com/api/v1/tasks/{task_id}"
    task_dict = api_requests.get(task_url)
    task = FullTask(**task_dict)
    if task.dict() != task_dict:
        warnings.warn("Task model in the library doesn't equal to the model returned by CVAT API.")
    return task


def _load_task_metadata(task_id: int) -> FullTaskMetadata:
    task_meta_url = f"https://cvat.piva-ai.com/api/v1/tasks/{task_id}/data/meta"
    meta_dict = api_requests.get(task_meta_url)
    meta = FullTaskMetadata(**meta_dict)
    if meta.dict() != meta_dict:
        warnings.warn(
            "Task Metadata model in the library doesn't equal to the model returned by CVAT API."
        )
    return meta


def load_project_data(
    project_id: int, *, return_dict: bool = False
) -> Tuple[Union[FullProject, dict], List[Union[Task, dict]]]:
    """Load project metadata from CVAT."""
    project = _load_project(project_id)
    tasks = [Task(**x.dict()) for x in project.tasks]  # get list of tasks in the project
    if return_dict:
        project = project.dict()
        tasks = [x.dict() for x in tasks]
    return project, tasks


def load_task_data(
    task_id: int, *, return_dict: bool = False
) -> Tuple[Union[FullTask, dict], List[Union[Job, dict]], Dict[int, Union[Frame, dict]]]:
    """Load task metadata from CVAT."""
    # load annotation data from CVAT
    task = _load_task(task_id)
    meta = _load_task_metadata(task_id)

    # get list of jobs in the task
    assert all(
        [len(x.jobs) == 1 for x in task.segments]
    ), "Unexpected CVAT data: one segment has multiple jobs."
    jobs = [
        Job(**job.dict(), start_frame=segment.start_frame, stop_frame=segment.stop_frame)
        for segment in task.segments
        for job in segment.jobs
    ]

    # get list of frames
    frame_ids_range = range(meta.start_frame, meta.stop_frame + 1)
    frames = {
        frame_id: Frame(  # frame id should be unique in the current task only
            id=x.name.split(".")[0],  # id should be unique across the whole dataset
            file_name=x.name.split("/")[-1],
            width=x.width,
            height=x.height,
            task_id=task_id,
            task_name=task.name,
        )
        for frame_id, x in zip(frame_ids_range, meta.frames)
    }

    # add job ids to the frames
    for job_data in jobs:
        for frame_id in range(job_data.start_frame, job_data.stop_frame + 1):
            assert (
                frame_id in frames
            ), f"Unexpected CVAT data: job ({job_data.id}) is missing a frame ({frame_id})."
            frames[frame_id].job_id = job_data.id
            frames[frame_id].status = job_data.status
    for frame_id, frame_data in frames.items():
        assert (
            "job_id" in frame_data
        ), f"Unexpected CVAT data: frame ({frame_id}) is missing job id."

    if return_dict:
        task = task.dict()
        jobs = [x.dict() for x in jobs]
        frames = {k: v.dict() for k, v in frames.items()}

    return task, jobs, frames


def load_annotations(
    job: Union[Job, dict], *, return_dict: bool = False
) -> Union[FullAnnotations, dict]:
    """Load annotations from a single job in CVAT."""
    if isinstance(job, Job):
        job_url = job.url
    else:
        assert "url" in job
        job_url = job["url"]
    annotations_dict = api_requests.get(os.path.join(job_url, "annotations"))
    annotations = FullAnnotations(**annotations_dict)
    if annotations.dict() != annotations_dict:
        warnings.warn(
            "Annotations model in the library doesn't equal to the model returned by CVAT API."
        )
    if return_dict:
        annotations = annotations.dict()
    return annotations


def download_images(task_id: int, output_path: str) -> List[str]:
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
