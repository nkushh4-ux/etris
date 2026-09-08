from datetime import UTC, datetime

import numpy as np

from cv.common.frame import FramePacket
from cv.detection.models import BoundingBox, VehicleDetection
from cv.tracking.models import TrackState, TrackStatus
from cv.tracking.simple import SimpleVehicleTracker


def make_detection(
    frame_index: int,
    x1: float = 10,
    y1: float = 20,
    x2: float = 110,
    y2: float = 120,
) -> VehicleDetection:
    frame = FramePacket(
        camera_id="CAM-001",
        frame_index=frame_index,
        timestamp=datetime.now(UTC),
        image=np.zeros((240, 320, 3), dtype=np.uint8),
    )

    return VehicleDetection(
        frame=frame,
        bbox=BoundingBox(x1, y1, x2, y2),
        class_name="car",
        confidence=0.91,
        detector_name="test",
        detector_version="1.0",
    )


def test_track_state_center():
    detection = make_detection(0)

    track = TrackState(
        track_id=1,
        camera_id="CAM-001",
        bbox=detection.bbox,
        class_name=detection.class_name,
        first_frame_index=0,
        last_frame_index=0,
        first_timestamp=detection.frame.timestamp,
        last_timestamp=detection.frame.timestamp,
        detection_confidence=detection.confidence,
        history=[detection.bbox],
    )

    assert track.center == (60.0, 70.0)


def test_track_state_update():
    first = make_detection(0)
    second = make_detection(
        1,
        x1=20,
        y1=30,
        x2=120,
        y2=130,
    )

    track = TrackState(
        track_id=1,
        camera_id="CAM-001",
        bbox=first.bbox,
        class_name=first.class_name,
        first_frame_index=0,
        last_frame_index=0,
        first_timestamp=first.frame.timestamp,
        last_timestamp=first.frame.timestamp,
        detection_confidence=first.confidence,
        history=[first.bbox],
    )

    track.update(second)

    assert track.track_id == 1
    assert track.last_frame_index == 1
    assert track.bbox == second.bbox
    assert track.hits == 2
    assert track.age == 2
    assert track.missed_frames == 0
    assert len(track.history) == 2


def test_simple_tracker_creates_track_ids():
    tracker = SimpleVehicleTracker()

    detections = [
        make_detection(0),
        make_detection(0, x1=150, x2=250),
    ]

    tracks = tracker.update(detections)

    assert len(tracks) == 2
    assert tracks[0].track_id == 1
    assert tracks[1].track_id == 2


def test_simple_tracker_assigns_new_id_to_new_vehicle():
    tracker = SimpleVehicleTracker()

    first = tracker.update([
        make_detection(0)
    ])

    second = tracker.update([
        make_detection(1),
        make_detection(
            1,
            x1=300,
            y1=200,
            x2=400,
            y2=300,
        ),
    ])

    assert first[0].track_id == 1
    assert second[0].track_id == 1
    assert second[1].track_id == 2

def test_track_starts_active():
    tracker = SimpleVehicleTracker()

    tracks = tracker.update([
        make_detection(0)
    ])

    assert tracks[0].status == TrackStatus.ACTIVE


def test_track_becomes_lost_after_missed_frame():
    tracker = SimpleVehicleTracker(
        max_missed_frames=2,
    )

    tracker.update([
        make_detection(0)
    ])

    tracker.update([])

    assert tracker._tracks[1].status == TrackStatus.LOST
    assert tracker._tracks[1].missed_frames == 1


def test_lost_track_becomes_active_after_recovery():
    tracker = SimpleVehicleTracker(
        max_missed_frames=2,
    )

    tracker.update([
        make_detection(0)
    ])

    tracker.update([])

    tracks = tracker.update([
        make_detection(
            2,
            x1=12,
            y1=22,
            x2=112,
            y2=122,
        )
    ])

    assert tracks[0].track_id == 1
    assert tracks[0].status == TrackStatus.ACTIVE


def test_expired_track_is_archived():
    tracker = SimpleVehicleTracker(
        max_missed_frames=1,
    )

    tracker.update([
        make_detection(0)
    ])

    tracker.update([])
    tracker.update([])

    assert 1 not in tracker._tracks
    assert 1 in tracker._completed_tracks
    assert tracker._completed_tracks[1].status == TrackStatus.EXPIRED


def test_single_detection_does_not_update_two_existing_tracks():
    tracker = SimpleVehicleTracker(minimum_iou=0.3, max_missed_frames=30)
    tracker.update([
        make_detection(0, x1=0, x2=100),
        make_detection(0, x1=10, x2=110),
    ])

    observed = make_detection(1, x1=5, x2=105)
    active = tracker.update([observed])

    assert [track.track_id for track in active] == [1]
    assert tracker._tracks[1].bbox == observed.bbox
    assert tracker._tracks[1].missed_frames == 0
    assert tracker._tracks[2].missed_frames == 1
    assert tracker._tracks[2].bbox != observed.bbox


def test_duplicate_exact_detection_does_not_create_duplicate_track():
    tracker = SimpleVehicleTracker()
    tracker.update([make_detection(0)])
    duplicate = make_detection(1, x1=12, y1=22, x2=112, y2=122)

    active = tracker.update([duplicate, duplicate])

    assert [track.track_id for track in active] == [1]
    assert list(tracker._tracks) == [1]


def test_two_detections_update_two_tracks_independently():
    tracker = SimpleVehicleTracker()
    tracker.update([
        make_detection(0, x1=0, x2=100),
        make_detection(0, x1=200, x2=300),
    ])
    first = make_detection(1, x1=2, x2=102)
    second = make_detection(1, x1=202, x2=302)

    active = tracker.update([first, second])

    assert [track.track_id for track in active] == [1, 2]
    assert tracker._tracks[1].bbox == first.bbox
    assert tracker._tracks[2].bbox == second.bbox
