from typing import Any, List, Optional, Union

from pydantic import BaseModel, root_validator

from cvat_utils import config

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
            original_url = values["url"].split("/api/")[1]
            values["url"] = f"{config.API_URL}/{original_url}"
        return values


class Frame(BaseModel):
    id: str
    frame_id: Union[int, str]
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


class FullLink(BaseModel):
    url: str
    count: int


class FullJobLink(FullLink):
    completed: Optional[int]
    validation: Optional[int]


class FullLabel(BaseModel):
    id: int
    name: str
    color: str
    attributes: List[dict]
    type: str
    sublabels: list
    project_id: int
    parent_id: Optional[int]
    has_parent: bool


class FullJob(BaseModel):
    id: int
    url: str
    task_id: int
    project_id: int
    assignee: Optional[dict]
    guide_id: Any  # not tested what the CVAT API returns
    dimension: str
    bug_tracker: str
    status: str
    stage: str
    state: str
    mode: str
    frame_count: int
    start_frame: int
    stop_frame: int
    data_chunk_size: int
    data_compressed_chunk_type: str
    created_date: str
    updated_date: str
    issues: FullLink
    labels: FullLink
    type: str
    organization: Any  # not tested what the CVAT API returns


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
    labels: FullLink
    jobs: FullJobLink
    data_chunk_size: int
    data_compressed_chunk_type: str
    data_original_chunk_type: str
    guide_id: Any  # not tested what the CVAT API returns
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
    related_files: int
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
    included_frames: Any  # not tested what the CVAT API returns


class FullProject(BaseModel):
    id: int
    url: str
    name: str
    labels: FullLink
    tasks: FullLink
    owner: Optional[dict]
    assignee: Optional[dict]
    guide_id: Any  # not tested what the CVAT API returns
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
