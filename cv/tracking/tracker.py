from abc import ABC, abstractmethod
from collections.abc import Sequence

from cv.common.frame import FramePacket
from cv.detection.models import VehicleDetection
from cv.tracking.models import TrackState


class VehicleTracker(ABC):
    """Interface implemented by vehicle tracking algorithms."""

    supports_lifecycle_events: bool = False

    @abstractmethod
    def update(
        self,
        detections: Sequence[VehicleDetection],
    ) -> Sequence[TrackState]:
        """Update tracker state using detections from one frame."""
        raise NotImplementedError

    def pop_expired_tracks(self) -> Sequence[TrackState]:
        """Return newly expired tracks once; trackers without events return none."""

        return ()

    def finalize_all(self) -> Sequence[TrackState]:
        """Remove and return tracks still retained by the tracker."""

        return ()

    def advance_without_detection(self, frame: FramePacket) -> Sequence[TrackState]:
        """Advance a deliberately unsampled frame without declaring a detector miss."""
        raise NotImplementedError
