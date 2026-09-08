from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class SightingStatus(str, Enum):
    CONFIRMED = "CONFIRMED"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"


@dataclass(frozen=True, slots=True)
class VehicleSighting:
    sighting_id: str
    plate_text: str
    camera_id: str
    timestamp: datetime
    latitude: float
    longitude: float
    plate_confidence: float
    source_track_id: int | None = None
    direction: str | None = None
    status: SightingStatus = SightingStatus.CONFIRMED
    source: str = "ANPR_FUSION"

    def __post_init__(self) -> None:
        if not self.sighting_id.strip() or not self.plate_text.strip():
            raise ValueError("sighting_id and plate_text must not be empty")
        if not self.camera_id.strip():
            raise ValueError("camera_id must not be empty")
        if not -90.0 <= self.latitude <= 90.0:
            raise ValueError("latitude must be between -90 and 90")
        if not -180.0 <= self.longitude <= 180.0:
            raise ValueError("longitude must be between -180 and 180")
        if not 0.0 <= self.plate_confidence <= 1.0:
            raise ValueError("plate_confidence must be between 0 and 1")

    @property
    def duplicate_key(self) -> tuple[str, str, datetime, int | None, str]:
        return (
            self.plate_text,
            self.camera_id,
            self.timestamp,
            self.source_track_id,
            self.source,
        )

