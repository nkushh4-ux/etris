from dataclasses import dataclass
from math import isfinite

from cv.detection.models import BoundingBox

Point = tuple[float, float]


def _validate_point(point: Point) -> None:
    if len(point) != 2 or any(not isfinite(x) or not 0 <= x <= 1 for x in point):
        raise ValueError("normalized coordinates must be finite and within 0..1")


@dataclass(frozen=True, slots=True)
class ApproachRegion:
    approach_id: str
    label: str
    traffic_polygon: tuple[Point, ...]
    queue_polygon: tuple[Point, ...]
    entry_line: tuple[Point, Point]
    exit_line: tuple[Point, Point] | None = None
    entry_direction: Point = (0.0, 1.0)

    def __post_init__(self) -> None:
        if not self.approach_id.strip() or not self.label.strip(): raise ValueError("approach identity must not be empty")
        if len(self.traffic_polygon) < 3 or len(self.queue_polygon) < 3: raise ValueError("polygons require at least three points")
        for point in self.traffic_polygon + self.queue_polygon + self.entry_line:
            _validate_point(point)
        if self.exit_line:
            for point in self.exit_line: _validate_point(point)
        if len(self.entry_direction) != 2 or any(not isfinite(x) for x in self.entry_direction) or self.entry_direction == (0, 0):
            raise ValueError("entry_direction must be a finite non-zero vector")


@dataclass(frozen=True, slots=True)
class TrafficStateConfig:
    intersection_id: str
    approaches: tuple[ApproachRegion, ...]
    arrival_window_s: float = 30
    stationary_motion_threshold: float = .012
    queue_stationary_min_s: float = 1.25
    controller_update_interval_s: float = 2
    track_expiry_s: float = 2

    def __post_init__(self) -> None:
        if not self.intersection_id.strip(): raise ValueError("intersection_id must not be empty")
        if not self.approaches: raise ValueError("at least one approach is required")
        if len({x.approach_id for x in self.approaches}) != len(self.approaches): raise ValueError("approach IDs must be unique")
        values = (self.arrival_window_s,self.stationary_motion_threshold,self.queue_stationary_min_s,
                  self.controller_update_interval_s,self.track_expiry_s)
        if any(not isfinite(x) or x <= 0 for x in values): raise ValueError("timing and motion configuration must be positive")


@dataclass(frozen=True, slots=True)
class TrackObservation:
    track_id: int
    bbox: BoundingBox
    class_name: str


@dataclass(frozen=True, slots=True)
class ExtractedApproachState:
    approach_id: str
    label: str
    active_vehicle_count: int
    queue_length: int
    image_space_occupancy: float
    average_waiting_time_s: float
    arrival_rate_vpm: float
    heavy_vehicle_count: int
    active_track_ids: tuple[int, ...]
    queued_track_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TrafficStateFrame:
    video_time_s: float
    frame_index: int
    approaches: tuple[ExtractedApproachState, ...]
    controller_update_due: bool
