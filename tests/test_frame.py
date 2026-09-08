from datetime import UTC, datetime

import numpy as np

from cv.common.frame import FramePacket


def test_frame_packet_stores_metadata_and_image():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    timestamp = datetime.now(UTC)

    frame = FramePacket(
        camera_id="CAM_001",
        frame_index=42,
        timestamp=timestamp,
        image=image,
    )

    assert frame.camera_id == "CAM_001"
    assert frame.frame_index == 42
    assert frame.timestamp == timestamp
    assert frame.image is image


def test_frame_packet_dimensions():
    image = np.zeros((720, 1280, 3), dtype=np.uint8)

    frame = FramePacket(
        camera_id="CAM_001",
        frame_index=0,
        timestamp=datetime.now(UTC),
        image=image,
    )

    assert frame.width == 1280
    assert frame.height == 720
    assert frame.channels == 3


def test_grayscale_frame_reports_one_channel():
    image = np.zeros((480, 640), dtype=np.uint8)

    frame = FramePacket(
        camera_id="CAM_001",
        frame_index=0,
        timestamp=datetime.now(UTC),
        image=image,
    )

    assert frame.channels == 1