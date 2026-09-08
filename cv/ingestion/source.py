from abc import ABC, abstractmethod
from collections.abc import Iterator

from cv.common.frame import FramePacket


class FrameSource(ABC):
    """Abstract source of video frames."""

    @abstractmethod
    def __iter__(self) -> Iterator[FramePacket]:
        """Yield frames from the source."""
        raise NotImplementedError