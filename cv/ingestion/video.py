from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import cv2

from cv.common.frame import FramePacket
from cv.ingestion.source import FrameSource


class VideoFileSource(FrameSource):
    """Read frames from a local video file."""

    def __init__(self, path: str | Path, camera_id: str) -> None:
        self.path = Path(path)
        self.camera_id = camera_id

        if not self.path.is_file():
            raise FileNotFoundError(f"Video file not found: {self.path}")

    def __iter__(self) -> Iterator[FramePacket]:
        capture = cv2.VideoCapture(str(self.path))

        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video: {self.path}")

        try:
            frame_index = 0

            while True:
                success, image = capture.read()

                if not success:
                    break

                yield FramePacket(
                    camera_id=self.camera_id,
                    frame_index=frame_index,
                    timestamp=datetime.now(UTC),
                    image=image,
                )

                frame_index += 1
        finally:
            capture.release()