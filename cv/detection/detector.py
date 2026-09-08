from abc import ABC, abstractmethod
from collections.abc import Sequence

from cv.common.frame import FramePacket
from cv.detection.models import VehicleDetection


class VehicleDetector(ABC):
    """Interface implemented by vehicle detection backends."""

    @abstractmethod
    def detect(self, frame: FramePacket) -> Sequence[VehicleDetection]:
        """Detect vehicles in one frame."""
        raise NotImplementedError