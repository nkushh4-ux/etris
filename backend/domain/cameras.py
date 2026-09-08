from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CameraNode:
    camera_id: str
    name: str
    latitude: float
    longitude: float
    road_name: str | None = None
    sector: str | None = None
    direction: str | None = None
    is_active: bool = True

    def __post_init__(self) -> None:
        if not self.camera_id.strip():
            raise ValueError("camera_id must not be empty")
        if not self.name.strip():
            raise ValueError("camera name must not be empty")
        if not -90.0 <= self.latitude <= 90.0:
            raise ValueError("latitude must be between -90 and 90")
        if not -180.0 <= self.longitude <= 180.0:
            raise ValueError("longitude must be between -180 and 180")

