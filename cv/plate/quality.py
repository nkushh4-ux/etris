from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from cv.plate.models import (
    PlateCandidate,
    PlateDetection,
    PlateQualityMetrics,
)

COMPONENTS = (
    "sharpness",
    "exposure",
    "contrast",
    "size",
    "detector_confidence",
)


@dataclass(frozen=True, slots=True)
class PlateQualityConfig:
    """Configurable V1 heuristic targets; these are not learned thresholds."""

    top_k: int = 3
    min_frame_gap: int = 2
    min_plate_width_px: int = 80
    fallback_min_plate_width_px: int | None = None
    min_plate_height_px: int = 8
    sharpness_target: float = 150.0
    exposure_target: float = 128.0
    exposure_tolerance: float = 128.0
    contrast_target: float = 64.0
    preferred_plate_width_px: int = 160
    preferred_plate_height_px: int = 48
    min_visible_fraction: float = 0.75
    weights: Mapping[str, float] = field(default_factory=lambda: {
        "sharpness": 0.35,
        "exposure": 0.15,
        "contrast": 0.15,
        "size": 0.20,
        "detector_confidence": 0.15,
    })

    def __post_init__(self) -> None:
        if self.fallback_min_plate_width_px is None:
            # Configurations created before fallback support retain their
            # original single-threshold behavior.
            object.__setattr__(
                self, "fallback_min_plate_width_px", self.min_plate_width_px
            )
        if self.top_k < 1:
            raise ValueError("top_k must be at least 1")
        if self.min_frame_gap < 0:
            raise ValueError("min_frame_gap must be non-negative")
        dimensions = (
            self.min_plate_width_px,
            self.fallback_min_plate_width_px,
            self.min_plate_height_px,
            self.preferred_plate_width_px,
            self.preferred_plate_height_px,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("plate dimensions must be positive")
        if self.fallback_min_plate_width_px > self.min_plate_width_px:
            raise ValueError(
                "fallback_min_plate_width_px must be less than or equal to "
                "min_plate_width_px"
            )
        targets = (
            self.sharpness_target,
            self.exposure_tolerance,
            self.contrast_target,
        )
        if any(value <= 0 for value in targets):
            raise ValueError("normalization targets must be positive")
        if not 0.0 < self.exposure_target <= 255.0:
            raise ValueError("exposure_target must be within (0, 255]")
        if not 0.0 < self.min_visible_fraction <= 1.0:
            raise ValueError("min_visible_fraction must be within (0, 1]")
        if set(self.weights) != set(COMPONENTS):
            raise ValueError(f"weights must contain exactly: {COMPONENTS}")
        if any(value < 0.0 for value in self.weights.values()):
            raise ValueError("component weights must be non-negative")
        if not np.isclose(sum(self.weights.values()), 1.0, atol=1e-6):
            raise ValueError("component weights must sum to approximately 1")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "PlateQualityConfig":
        """Build and validate configuration from the YAML subsection."""

        return cls(**dict(values))


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class PlateQualityAssessor:
    """Create and score a plate crop without retaining the source frame."""

    def __init__(self, config: PlateQualityConfig | None = None) -> None:
        self.config = config or PlateQualityConfig()

    def assess(self, detection: PlateDetection) -> PlateCandidate | None:
        image = detection.frame.image
        if image.size == 0 or image.ndim not in (2, 3):
            return None

        height, width = image.shape[:2]
        box = detection.bbox
        requested_area = box.width * box.height
        if requested_area <= 0.0:
            return None

        x1 = max(0, min(width, int(np.floor(box.x1))))
        y1 = max(0, min(height, int(np.floor(box.y1))))
        x2 = max(0, min(width, int(np.ceil(box.x2))))
        y2 = max(0, min(height, int(np.ceil(box.y2))))
        crop_width = x2 - x1
        crop_height = y2 - y1
        if (
            crop_width < self.config.fallback_min_plate_width_px
            or crop_height < self.config.min_plate_height_px
        ):
            return None

        visible_fraction = _clamp(
            (crop_width * crop_height) / requested_area
        )
        if visible_fraction < self.config.min_visible_fraction:
            return None

        crop = image[y1:y2, x1:x2].copy()
        if crop.size == 0:
            return None

        gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_intensity = float(gray.mean())
        intensity_stddev = float(gray.std())

        sharpness = _clamp(laplacian_variance / self.config.sharpness_target)
        exposure = _clamp(
            1.0
            - abs(mean_intensity - self.config.exposure_target)
            / self.config.exposure_tolerance
        )
        contrast = _clamp(intensity_stddev / self.config.contrast_target)
        size = _clamp(min(
            crop_width / self.config.preferred_plate_width_px,
            crop_height / self.config.preferred_plate_height_px,
        ))
        confidence = _clamp(detection.confidence)
        clipping = visible_fraction

        components = {
            "sharpness": sharpness,
            "exposure": exposure,
            "contrast": contrast,
            "size": size,
            "detector_confidence": confidence,
        }
        base_score = sum(
            self.config.weights[name] * value
            for name, value in components.items()
        )
        # Partial clipping is a soft penalty; severe clipping is rejected above.
        quality_score = _clamp(base_score * (0.5 + 0.5 * clipping))

        metrics = PlateQualityMetrics(
            **components,
            clipping=clipping,
            laplacian_variance=laplacian_variance,
            mean_intensity=mean_intensity,
            intensity_stddev=intensity_stddev,
            crop_width_px=crop_width,
            crop_height_px=crop_height,
            visible_fraction=visible_fraction,
        )
        crop.setflags(write=False)
        return PlateCandidate(
            camera_id=detection.frame.camera_id,
            track_id=detection.track_id,
            frame_index=detection.frame.frame_index,
            timestamp=detection.frame.timestamp,
            bbox=detection.bbox,
            detector_confidence=confidence,
            metrics=metrics,
            quality_score=quality_score,
            crop=crop,
            selection_tier=(
                "PRIMARY"
                if crop_width >= self.config.min_plate_width_px
                else "FALLBACK"
            ),
        )
