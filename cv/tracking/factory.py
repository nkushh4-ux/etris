from cv.tracking.simple import SimpleVehicleTracker
from cv.tracking.tracker import VehicleTracker
from cv.tracking.types import TrackingAlgorithm


def create_tracker(
    algorithm: TrackingAlgorithm,
    *,
    minimum_iou: float = 0.3,
    max_missed_frames: int = 30,
) -> VehicleTracker:
    """Create a configured vehicle tracker."""

    if algorithm == TrackingAlgorithm.SIMPLE_IOU:
        return SimpleVehicleTracker(
            minimum_iou=minimum_iou,
            max_missed_frames=max_missed_frames,
        )

    if algorithm in {
        TrackingAlgorithm.BYTE_TRACK,
        TrackingAlgorithm.BOT_SORT,
    }:
        raise NotImplementedError(
            f"Production tracker backend not implemented: {algorithm.value}"
        )

    raise ValueError(
        f"Unsupported tracking algorithm: {algorithm}"
    )