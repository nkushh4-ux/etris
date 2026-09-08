from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np

from cv.common.frame import FramePacket
from cv.detection.models import BoundingBox
from scripts.run_anpr_diagnostics import (
    bbox_metrics,
    compare_track_pairs,
    quality_rejection_reasons,
)


def quality_config():
    return SimpleNamespace(
        min_plate_width_px=80,
        min_plate_height_px=8,
        min_visible_fraction=.75,
    )


def detection(box):
    packet = FramePacket("A", 1, datetime.now(UTC),
                         np.zeros((100, 200, 3), dtype=np.uint8))
    return SimpleNamespace(frame=packet, bbox=box)


def test_quality_rejection_reason_aggregation_is_diagnostic_only():
    reasons = quality_rejection_reasons(
        detection(BoundingBox(10, 10, 30, 15)), quality_config()
    )
    assert reasons == ("width_below_minimum", "height_below_minimum")


def test_bbox_comparison_metrics():
    iou, distance = bbox_metrics([0, 0, 10, 10], [5, 0, 15, 10])
    assert iou == 1 / 3
    assert distance == 5


def test_overlap_comparison_reports_coexistence_and_nearest_frames():
    histories = {
        ("A", 21): [
            {"frame_index": 258, "bbox": [0, 0, 10, 10]},
            {"frame_index": 263, "bbox": [2, 0, 12, 10]},
        ],
        ("A", 23): [
            {"frame_index": 263, "bbox": [3, 0, 13, 10]},
            {"frame_index": 275, "bbox": [8, 0, 18, 10]},
        ],
    }
    result = compare_track_pairs(histories)
    assert len(result) == 1
    assert result[0]["track_a"] == 21 and result[0]["track_b"] == 23
    assert result[0]["coexist_in_same_observed_frame"] is True
    assert result[0]["nearest_observed_frames"]["frame_distance"] == 0
