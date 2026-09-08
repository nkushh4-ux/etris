from pathlib import Path

import cv2
import numpy as np
import pytest

from cv.ingestion.video import VideoFileSource


def create_test_video(path: Path, frame_count: int = 3) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (320, 240),
    )

    if not writer.isOpened():
        raise RuntimeError("Unable to create test video")

    try:
        for _ in range(frame_count):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def test_video_file_source_yields_frames(tmp_path):
    video_path = tmp_path / "test.mp4"
    create_test_video(video_path, frame_count=3)

    source = VideoFileSource(video_path, camera_id="CAM_001")
    frames = list(source)

    assert len(frames) == 3
    assert frames[0].camera_id == "CAM_001"
    assert frames[0].frame_index == 0
    assert frames[1].frame_index == 1
    assert frames[2].frame_index == 2


def test_video_file_source_preserves_frame_dimensions(tmp_path):
    video_path = tmp_path / "test.mp4"
    create_test_video(video_path)

    source = VideoFileSource(video_path, camera_id="CAM_001")
    frame = next(iter(source))

    assert frame.width == 320
    assert frame.height == 240
    assert frame.channels == 3


def test_video_file_source_rejects_missing_file(tmp_path):
    missing_path = tmp_path / "missing.mp4"

    with pytest.raises(FileNotFoundError):
        VideoFileSource(missing_path, camera_id="CAM_001")