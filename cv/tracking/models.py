from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from cv.detection.models import BoundingBox, VehicleDetection


class TrackStatus(str, Enum):
    """Lifecycle state of a vehicle track."""

    ACTIVE = "active"
    LOST = "lost"
    EXPIRED = "expired"


@dataclass(slots=True)
class TrackState:
    """Current state of one tracked vehicle."""

    track_id: int
    camera_id: str
    bbox: BoundingBox
    class_name: str

    first_frame_index: int
    last_frame_index: int

    first_timestamp: datetime
    last_timestamp: datetime

    detection_confidence: float

    age: int = 1
    hits: int = 1
    missed_frames: int = 0

    history: list[BoundingBox] = field(default_factory=list)

    status: TrackStatus = TrackStatus.ACTIVE

    def update(self, detection: VehicleDetection) -> None:
        """Update this track with a new detection."""

        self.bbox = detection.bbox
        self.class_name = detection.class_name
        self.last_frame_index = detection.frame.frame_index
        self.last_timestamp = detection.frame.timestamp
        self.detection_confidence = detection.confidence

        self.age += 1
        self.hits += 1
        self.missed_frames = 0
        self.status = TrackStatus.ACTIVE

        self.history.append(detection.bbox)

    @property
    def center(self) -> tuple[float, float]:
        """Return the current vehicle center."""

        return self.bbox.center