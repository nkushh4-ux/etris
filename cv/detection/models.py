from dataclasses import dataclass

from cv.common.frame import FramePacket


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Axis-aligned bounding box in image coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.x1 + self.x2) / 2.0,
            (self.y1 + self.y2) / 2.0,
        )


@dataclass(frozen=True, slots=True)
class VehicleDetection:
    """One vehicle detection produced from a frame."""

    frame: FramePacket
    bbox: BoundingBox
    class_name: str
    confidence: float
    detector_name: str
    detector_version: str