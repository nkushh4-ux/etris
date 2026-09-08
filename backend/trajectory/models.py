from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class PlausibilityStatus(str, Enum):
    PLAUSIBLE = "PLAUSIBLE"
    TOO_FAST = "TOO_FAST"
    TOO_SLOW = "TOO_SLOW"
    NO_TOPOLOGY = "NO_TOPOLOGY"


class TrajectoryStatus(str, Enum):
    EMPTY = "EMPTY"
    COMPLETE = "COMPLETE"
    HAS_IMPLAUSIBLE_TRANSITIONS = "HAS_IMPLAUSIBLE_TRANSITIONS"
    INCOMPLETE_TOPOLOGY = "INCOMPLETE_TOPOLOGY"


@dataclass(frozen=True, slots=True)
class CameraEdge:
    source_camera_id: str
    target_camera_id: str
    distance_m: float
    min_travel_seconds: float
    max_travel_seconds: float

    def __post_init__(self) -> None:
        if not self.source_camera_id or not self.target_camera_id:
            raise ValueError("edge camera IDs must not be empty")
        if self.distance_m < 0 or self.min_travel_seconds < 0:
            raise ValueError("edge distance and minimum travel time must be non-negative")
        if self.max_travel_seconds < self.min_travel_seconds:
            raise ValueError("maximum travel time must be >= minimum travel time")


@dataclass(frozen=True, slots=True)
class TrajectoryPoint:
    camera_id: str
    camera_name: str
    latitude: float
    longitude: float
    timestamp: datetime
    confidence: float
    direction: str | None = None
    sighting_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id, "camera_name": self.camera_name,
            "lat": self.latitude, "lon": self.longitude,
            "timestamp": self.timestamp.isoformat(), "confidence": self.confidence,
            "direction": self.direction, "sighting_id": self.sighting_id,
        }


@dataclass(frozen=True, slots=True)
class TrajectorySegment:
    source_camera_id: str
    target_camera_id: str
    start_timestamp: datetime
    end_timestamp: datetime
    travel_time_seconds: float
    distance_m: float | None
    estimated_speed_kmh: float | None
    plausibility_status: PlausibilityStatus
    direction: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "from": self.source_camera_id, "to": self.target_camera_id,
            "start_timestamp": self.start_timestamp.isoformat(),
            "end_timestamp": self.end_timestamp.isoformat(),
            "travel_time_s": self.travel_time_seconds,
            "distance_m": self.distance_m, "speed_kmh": self.estimated_speed_kmh,
            "status": self.plausibility_status.value,
            "direction": self.direction,
        }


@dataclass(frozen=True, slots=True)
class VehicleTrajectory:
    plate_text: str
    points: tuple[TrajectoryPoint, ...]
    segments: tuple[TrajectorySegment, ...]
    start_time: datetime | None
    end_time: datetime | None
    total_distance_m: float
    total_duration_seconds: float
    average_speed_kmh: float | None
    camera_count: int
    status: TrajectoryStatus
    diagnostics: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "plate": self.plate_text,
            "points": [point.to_dict() for point in self.points],
            "segments": [segment.to_dict() for segment in self.segments],
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_distance_m": self.total_distance_m,
            "total_duration_s": self.total_duration_seconds,
            "average_speed_kmh": self.average_speed_kmh,
            "camera_count": self.camera_count, "status": self.status.value,
            "diagnostics": list(self.diagnostics),
        }
