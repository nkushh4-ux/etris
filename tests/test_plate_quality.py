from datetime import UTC, datetime

import cv2
import numpy as np
import pytest

from cv.common.frame import FramePacket
from cv.detection.models import BoundingBox
from cv.plate.models import PlateDetection
from cv.plate.quality import PlateQualityAssessor, PlateQualityConfig


def detection(image, bbox=None, confidence=0.9):
    frame = FramePacket("CAM-1", 4, datetime.now(UTC), image)
    height, width = image.shape[:2]
    return PlateDetection(
        frame=frame,
        track_id=7,
        bbox=bbox or BoundingBox(0, 0, width, height),
        confidence=confidence,
        detector_name="test",
        detector_version="1",
    )


def assessor(**overrides):
    values = {
        "min_plate_width_px": 4,
        "min_plate_height_px": 4,
        "preferred_plate_width_px": 40,
        "preferred_plate_height_px": 12,
    }
    values.update(overrides)
    return PlateQualityAssessor(PlateQualityConfig(**values))


def test_component_scores_remain_normalized():
    image = np.random.default_rng(1).integers(0, 256, (20, 60, 3), dtype=np.uint8)
    candidate = assessor().assess(detection(image, confidence=2.0))
    values = (
        candidate.metrics.sharpness,
        candidate.metrics.exposure,
        candidate.metrics.contrast,
        candidate.metrics.size,
        candidate.metrics.detector_confidence,
        candidate.metrics.clipping,
        candidate.quality_score,
    )
    assert all(0.0 <= value <= 1.0 for value in values)


def test_sharp_scores_above_blurred():
    sharp = np.indices((24, 64)).sum(axis=0) % 2 * 255
    sharp = sharp.astype(np.uint8)
    blurred = cv2.GaussianBlur(sharp, (15, 15), 0)
    scorer = assessor()
    assert scorer.assess(detection(sharp)).metrics.sharpness > scorer.assess(detection(blurred)).metrics.sharpness


@pytest.mark.parametrize("bad_level", [0, 255])
def test_reasonable_exposure_scores_above_extremes(bad_level):
    scorer = assessor()
    normal = np.full((20, 60), 128, dtype=np.uint8)
    extreme = np.full((20, 60), bad_level, dtype=np.uint8)
    assert scorer.assess(detection(normal)).metrics.exposure > scorer.assess(detection(extreme)).metrics.exposure


def test_adequate_size_scores_above_tiny_plate():
    scorer = assessor(min_plate_width_px=2, min_plate_height_px=2)
    large = np.full((20, 60), 128, dtype=np.uint8)
    tiny = np.full((4, 8), 128, dtype=np.uint8)
    assert scorer.assess(detection(large)).metrics.size > scorer.assess(detection(tiny)).metrics.size


@pytest.mark.parametrize(
    "image,bbox",
    [
        (np.empty((0, 0), dtype=np.uint8), BoundingBox(0, 0, 1, 1)),
        (np.zeros((20, 20), dtype=np.uint8), BoundingBox(5, 5, 5, 10)),
        (np.zeros((20, 20), dtype=np.uint8), BoundingBox(0, 0, 2, 2)),
        (np.zeros((20, 20), dtype=np.uint8), BoundingBox(-100, -100, 10, 10)),
    ],
)
def test_invalid_or_unusable_crops_are_rejected(image, bbox):
    assert assessor().assess(detection(image, bbox)) is None


def test_candidate_owns_crop_without_retaining_frame():
    image = np.full((20, 60), 128, dtype=np.uint8)
    source = detection(image)
    candidate = assessor().assess(source)
    assert not hasattr(candidate, "frame")
    assert not hasattr(candidate, "detection")
    assert not np.shares_memory(candidate.crop, image)
    assert not candidate.crop.flags.writeable


def test_configuration_validation():
    with pytest.raises(ValueError):
        PlateQualityConfig(top_k=0)
    with pytest.raises(ValueError):
        PlateQualityConfig(weights={name: 0.1 for name in (
            "sharpness", "exposure", "contrast", "size", "detector_confidence"
        )})
    with pytest.raises(ValueError):
        PlateQualityConfig(
            min_plate_width_px=80,
            fallback_min_plate_width_px=81,
        )


def test_primary_and_fallback_width_eligibility():
    scorer = assessor(
        min_plate_width_px=80,
        fallback_min_plate_width_px=70,
    )
    primary = scorer.assess(detection(np.full((12, 80), 128, dtype=np.uint8)))
    fallback = scorer.assess(detection(np.full((12, 70), 128, dtype=np.uint8)))

    assert primary.selection_tier == "PRIMARY"
    assert fallback.selection_tier == "FALLBACK"
    assert scorer.assess(detection(np.full((12, 69), 128, dtype=np.uint8))) is None


def test_legacy_config_keeps_single_width_threshold():
    config = PlateQualityConfig(min_plate_width_px=70)
    assert config.fallback_min_plate_width_px == 70
