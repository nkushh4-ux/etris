from collections.abc import Sequence
from datetime import UTC, datetime

import numpy as np

from cv.common.frame import FramePacket
from cv.detection.detector import VehicleDetector
from cv.detection.models import BoundingBox, VehicleDetection
from cv.pipeline.camera import CameraPipeline
from cv.plate.models import PlateDetection
from cv.plate.quality import PlateQualityAssessor, PlateQualityConfig
from cv.plate.selection import PlateBestFrameSelector
from cv.tracking.models import TrackState
from cv.tracking.tracker import VehicleTracker


class FakeDetector(VehicleDetector):
    """Detector used for pipeline tests."""

    def detect(
        self,
        frame: FramePacket,
    ) -> Sequence[VehicleDetection]:
        return [
            VehicleDetection(
                frame=frame,
                bbox=BoundingBox(
                    10,
                    20,
                    110,
                    120,
                ),
                class_name="car",
                confidence=0.95,
                detector_name="fake",
                detector_version="1.0",
            )
        ]


class FakeTracker(VehicleTracker):
    """Tracker used for pipeline tests."""

    def update(
        self,
        detections: Sequence[VehicleDetection],
    ) -> Sequence[TrackState]:
        detection = detections[0]

        return [
            TrackState(
                track_id=42,
                camera_id=detection.frame.camera_id,
                bbox=detection.bbox,
                class_name=detection.class_name,
                first_frame_index=detection.frame.frame_index,
                last_frame_index=detection.frame.frame_index,
                first_timestamp=detection.frame.timestamp,
                last_timestamp=detection.frame.timestamp,
                detection_confidence=detection.confidence,
                history=[detection.bbox],
            )
        ]


class FakePlateDetector:
    def detect_in_frame(self, track, image, frame_packet):
        return [
            PlateDetection(
                frame=frame_packet,
                track_id=track.track_id,
                bbox=BoundingBox(20, 30, 80, 50),
                confidence=0.9,
                detector_name="fake-plate",
                detector_version="1.0",
            )
        ]


def make_frame() -> FramePacket:
    return FramePacket(
        camera_id="CAM-001",
        frame_index=0,
        timestamp=datetime.now(UTC),
        image=np.zeros(
            (240, 320, 3),
            dtype=np.uint8,
        ),
    )


def test_camera_pipeline_connects_detector_to_tracker():
    pipeline = CameraPipeline(
        detector=FakeDetector(),
        tracker=FakeTracker(),
    )

    tracks = pipeline.process_frame(make_frame())

    assert len(tracks) == 1
    assert tracks[0].track_id == 42
    assert tracks[0].camera_id == "CAM-001"
    assert tracks[0].class_name == "car"


def test_camera_pipeline_preserves_detection_provenance():
    pipeline = CameraPipeline(
        detector=FakeDetector(),
        tracker=FakeTracker(),
    )

    frame = make_frame()
    tracks = pipeline.process_frame(frame)

    assert tracks[0].first_frame_index == frame.frame_index
    assert tracks[0].last_frame_index == frame.frame_index
    assert tracks[0].detection_confidence == 0.95


def test_camera_pipeline_offers_scored_plate_candidate():
    selector = PlateBestFrameSelector()
    pipeline = CameraPipeline(
        detector=FakeDetector(),
        tracker=FakeTracker(),
        plate_detector=FakePlateDetector(),
        quality_assessor=PlateQualityAssessor(PlateQualityConfig(
            min_plate_width_px=4,
            min_plate_height_px=4,
        )),
        best_frame_selector=selector,
    )

    pipeline.process_frame(make_frame())

    retained = selector.get("CAM-001", 42)
    assert len(retained) == 1
    assert retained[0].crop.shape == (20, 60, 3)
