from enum import Enum


class TrackingAlgorithm(str, Enum):
    """Supported vehicle tracking algorithms."""

    SIMPLE_IOU = "simple_iou"
    BYTE_TRACK = "byte_track"
    BOT_SORT = "bot_sort"