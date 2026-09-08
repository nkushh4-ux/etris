from dataclasses import dataclass
from datetime import datetime

import numpy as np

from cv.common.frame import FramePacket
from cv.detection.models import BoundingBox
from cv.plate.types import PlateCategory


@dataclass(frozen=True, slots=True)
class PlateDetection:
    """A license plate detected within a tracked vehicle."""

    frame: FramePacket
    track_id: int
    bbox: BoundingBox
    confidence: float

    detector_name: str
    detector_version: str

    category: PlateCategory = PlateCategory.UNSUPPORTED
    category_confidence: float = 0.0


@dataclass(frozen=True, slots=True)
class PlateQualityMetrics:
    """Normalized quality components and their raw measurements."""

    sharpness: float
    exposure: float
    contrast: float
    size: float
    detector_confidence: float
    clipping: float

    laplacian_variance: float
    mean_intensity: float
    intensity_stddev: float
    crop_width_px: int
    crop_height_px: int
    visible_fraction: float


@dataclass(frozen=True, slots=True)
class PlateCandidate:
    """Lightweight retained plate crop, independent of its source frame."""

    camera_id: str
    track_id: int
    frame_index: int
    timestamp: datetime
    bbox: BoundingBox
    detector_confidence: float
    metrics: PlateQualityMetrics
    quality_score: float
    crop: np.ndarray
    selection_tier: str = "PRIMARY"
