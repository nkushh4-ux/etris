from datetime import UTC, datetime

import numpy as np

from cv.common.frame import FramePacket
from cv.detection.models import BoundingBox, VehicleDetection


def make_frame() -> FramePacket:
    return FramePacket(
        camera_id="CAM_001",
        frame_index=10,
        timestamp=datetime.now(UTC),
        image=np.zeros((480, 640, 3), dtype=np.uint8),
    )


def test_bounding_box_dimensions():
    bbox = BoundingBox(
        x1=100,
        y1=50,
        x2=300,
        y2=250,
    )

    assert bbox.width == 200
    assert bbox.height == 200
    assert bbox.center == (200, 150)


def test_bounding_box_rejects_negative_dimensions():
    bbox = BoundingBox(
        x1=300,
        y1=250,
        x2=100,
        y2=50,
    )

    assert bbox.width == 0
    assert bbox.height == 0


def test_vehicle_detection_preserves_provenance():
    frame = make_frame()

    detection = VehicleDetection(
        frame=frame,
        bbox=BoundingBox(100, 50, 300, 250),
        class_name="car",
        confidence=0.91,
        detector_name="test-detector",
        detector_version="0.1.0",
    )

    assert detection.frame is frame
    assert detection.class_name == "car"
    assert detection.confidence == 0.91
    assert detection.detector_name == "test-detector"
    assert detection.detector_version == "0.1.0"