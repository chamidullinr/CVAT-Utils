from typing import Any, List, Optional

from pydantic import BaseModel, root_validator

from cvat_utils.config import API_URL

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
            # replace IP addr with base url
            path = values["url"].split("/api/")[1]
            values["url"] = f"{API_URL}/{path}"
        return values


class Frame(BaseModel):
    id: str
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
    type: str
    sublabels: list
    has_parent: bool


class FullJob(BaseModel):
    id: int
    url: str
    status: str
    assignee: Optional[dict]
    status: str
    stage: str
    state: str


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
    organization: Any  # not tested what the CVAT API returns
    target_storage: Any  # not tested what the CVAT API returns
    source_storage: Any  # not tested what the CVAT API returns


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
    deleted_frames: list  # not tested what the CVAT API returns


class FullProject(BaseModel):
    id: int
    url: str
    name: str
    labels: List[FullLabel]
    tasks: List[int]
    owner: Optional[dict]
    assignee: Optional[dict]
    bug_tracker: str
    created_date: str
    updated_date: str
    status: str
    dimension: str
    organization: Any  # not tested what the CVAT API returns
    target_storage: Any  # not tested what the CVAT API returns
    source_storage: Any  # not tested what the CVAT API returns
    task_subsets: list  # not tested what the CVAT API returns


class FullAnnotationBase(BaseModel):
    id: int
    frame: int
    label_id: int
    group: Optional[int]
    source: str
    attributes: List[dict]


class FullTag(FullAnnotationBase):
    pass


class FullShape(FullAnnotationBase):
    type: str
    occluded: bool
    outside: bool
    z_order: int
    rotation: float
    points: list
    elements: list  # not tested what the CVAT API returns


class FullTrackShape(BaseModel):
    id: int
    frame: int
    attributes: List[dict]
    type: str
    occluded: bool
    outside: bool
    z_order: int
    rotation: float
    points: list


class FullTrack(FullAnnotationBase):
    shapes: List[FullTrackShape]


class FullAnnotations(BaseModel):
    version: int
    tags: List[FullTag]
    shapes: List[FullShape]
    tracks: List[FullTrack]
