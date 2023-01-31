from typing import Any, List, Optional

from pydantic import BaseModel, root_validator

"""
Custom CVAT models.
"""


class Task(BaseModel):
    id: int
    name: str
    status: str


class Job(BaseModel):
    id: int
    url: str
    status: str
    start_frame: int
    stop_frame: int

    @root_validator(pre=True)
    def set_variables(cls, values: dict) -> dict:
        """Preprocess class variables."""
        if "url" in values:
            values["url"] = values["url"].replace("http://", "https://")
        return values


class Frame(BaseModel):
    id: int
    file_name: str
    width: int
    height: int
    task_id: int
    task_name: str
    job_id: Optional[int]
    status: Optional[str]


"""
CVAT API models.
"""


class FullLabel(BaseModel):
    id: int
    name: str
    color: str
    attributes: List[dict]


class FullJob(BaseModel):
    id: int
    url: str
    status: str
    assignee: Optional[dict]
    reviewer: Optional[dict]


class FullSegment(BaseModel):
    start_frame: int
    stop_frame: int
    jobs: List[FullJob]


class FullTask(BaseModel):
    id: int
    url: str
    name: str
    project_id: int
    mode: str
    owner: Optional[dict]
    assignee: Optional[dict]
    bug_tracker: str
    created_date: str
    updated_date: str
    overlap: int
    segment_size: int
    status: str
    labels: List[FullLabel]
    segments: List[FullSegment]
    data_chunk_size: int
    data_compressed_chunk_type: str
    data_original_chunk_type: str
    size: int
    image_quality: int
    data: int
    dimension: str
    subset: str


class FullFrame(BaseModel):
    width: int
    height: int
    name: str
    has_related_context: bool


class FullTaskMetadata(BaseModel):
    chunk_size: int
    size: int
    image_quality: int
    start_frame: int
    stop_frame: int
    frame_filter: str
    frames: List[FullFrame]


class FullProject(BaseModel):
    id: int
    url: str
    name: str
    labels: List[FullLabel]
    tasks: List[FullTask]
    owner: Optional[dict]
    assignee: Optional[dict]
    bug_tracker: str
    created_date: str
    updated_date: str
    status: str
    training_project: Optional[Any]
    dimension: str


class FullTag(BaseModel):
    id: int
    frame: int
    label_id: int
    group: int
    source: str
    attributes: List[dict]


class FullShape(BaseModel):
    id: int
    frame: int
    label_id: int
    group: int
    source: str
    attributes: List[dict]
    type: str
    occluded: bool
    z_order: int
    points: list


class FullAnnotations(BaseModel):
    version: int
    tags: List[FullTag]
    shapes: List[FullShape]
    tracks: list  # TODO - add FullTrack
