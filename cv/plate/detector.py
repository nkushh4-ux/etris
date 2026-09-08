from abc import ABC, abstractmethod
from collections.abc import Sequence

from cv.plate.models import PlateDetection
from cv.tracking.models import TrackState


class PlateDetector(ABC):
    """Interface for license plate detection backends."""

    @abstractmethod
    def detect(
        self,
        track: TrackState,
    ) -> Sequence[PlateDetection]:
        """Detect license plates belonging to a vehicle track."""
        raise NotImplementedError

    def detect_batch_in_frame(self, tracks, image, frame_packet):
        """Detect scheduled ROIs together when supported by the backend."""
        detections = []
        for track in tracks:
            detections.extend(self.detect_in_frame(track, image, frame_packet))
        return detections
