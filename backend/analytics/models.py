from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class CongestionLevel(str, Enum):
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    FREE_FLOW = "FREE_FLOW"
    MODERATE = "MODERATE"
    CONGESTED = "CONGESTED"
    SEVERE = "SEVERE"


class Serializable:
    def to_dict(self) -> dict[str, Any]:
        def convert(value):
            if isinstance(value, datetime): return value.isoformat()
            if isinstance(value, Enum): return value.value
            if isinstance(value, tuple): return [convert(x) for x in value]
            if isinstance(value, dict): return {k: convert(v) for k, v in value.items()}
            return value
        return {key: convert(value) for key, value in asdict(self).items()}


@dataclass(frozen=True, slots=True)
class CameraVolume(Serializable):
    camera_id: str
    vehicle_count: int
    unique_vehicle_count: int
    vehicles_per_hour: float
    window_start: datetime
    window_end: datetime


@dataclass(frozen=True, slots=True)
class ODFlow(Serializable):
    origin_camera: str
    destination_camera: str
    vehicle_count: int


@dataclass(frozen=True, slots=True)
class SegmentAnalytics(Serializable):
    source_camera: str
    target_camera: str
    vehicle_count: int
    mean_travel_time_s: float
    median_travel_time_s: float
    mean_speed_kmh: float
    median_speed_kmh: float


@dataclass(frozen=True, slots=True)
class CongestionResult(Serializable):
    source_camera: str
    target_camera: str
    sample_count: int
    baseline_travel_time_s: float
    observed_median_travel_time_s: float | None
    travel_time_ratio: float | None
    classification: CongestionLevel


@dataclass(frozen=True, slots=True)
class HeatmapPoint(Serializable):
    camera_id: str
    latitude: float
    longitude: float
    vehicle_count: int
    normalized_intensity: float


@dataclass(frozen=True, slots=True)
class RouteDensity(Serializable):
    source: str
    target: str
    vehicle_count: int
    normalized_density: float


@dataclass(frozen=True, slots=True)
class NetworkSummary(Serializable):
    window_start: datetime
    window_end: datetime
    total_sightings: int
    unique_vehicles: int
    active_cameras: int
    inter_camera_movements: int
    busiest_camera: str | None
    busiest_camera_count: int
    busiest_segment: str | None
    busiest_segment_count: int
    average_valid_speed_kmh: float | None
    median_valid_speed_kmh: float | None
    free_flow_segments: int
    moderate_segments: int
    congested_segments: int
    severe_segments: int
