from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass(frozen=True, slots=True)
class FramePacket:
    """A video frame together with its source and timing metadata."""

    camera_id: str
    frame_index: int
    timestamp: datetime
    image: np.ndarray

    @property
    def height(self) -> int:
        """Return frame height in pixels."""
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        """Return frame width in pixels."""
        return int(self.image.shape[1])

    @property
    def channels(self) -> int:
        """Return the number of image channels."""
        if self.image.ndim != 3:
            return 1

        return int(self.image.shape[2])