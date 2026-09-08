import pytest

from cv.tracking.factory import create_tracker
from cv.tracking.simple import SimpleVehicleTracker
from cv.tracking.types import TrackingAlgorithm


def test_factory_creates_simple_tracker():
    tracker = create_tracker(
        TrackingAlgorithm.SIMPLE_IOU,
    )

    assert isinstance(tracker, SimpleVehicleTracker)


def test_factory_passes_simple_tracker_configuration():
    tracker = create_tracker(
        TrackingAlgorithm.SIMPLE_IOU,
        minimum_iou=0.5,
        max_missed_frames=10,
    )

    assert isinstance(tracker, SimpleVehicleTracker)
    assert tracker.minimum_iou == 0.5
    assert tracker.max_missed_frames == 10


def test_factory_rejects_unimplemented_byte_track():
    with pytest.raises(NotImplementedError):
        create_tracker(
            TrackingAlgorithm.BYTE_TRACK,
        )


def test_factory_rejects_unimplemented_bot_sort():
    with pytest.raises(NotImplementedError):
        create_tracker(
            TrackingAlgorithm.BOT_SORT,
        )