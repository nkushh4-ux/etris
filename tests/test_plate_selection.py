from datetime import UTC, datetime

import numpy as np

from cv.detection.models import BoundingBox
from cv.plate.models import PlateCandidate, PlateQualityMetrics
from cv.plate.selection import PlateBestFrameSelector

METRICS = PlateQualityMetrics(
    sharpness=0.5, exposure=0.5, contrast=0.5, size=0.5,
    detector_confidence=0.5, clipping=1.0, laplacian_variance=10.0,
    mean_intensity=128.0, intensity_stddev=20.0, crop_width_px=20,
    crop_height_px=8, visible_fraction=1.0,
)


def candidate(frame, score, confidence=0.9, camera="A", track=1, tier="PRIMARY"):
    crop = np.zeros((8, 20), dtype=np.uint8)
    crop.setflags(write=False)
    return PlateCandidate(
        camera, track, frame, datetime.now(UTC), BoundingBox(0, 0, 20, 8),
        confidence, METRICS, score, crop, tier,
    )


def test_top_k_is_bounded_and_better_candidate_replaces_worst():
    selector = PlateBestFrameSelector(top_k=2, min_frame_gap=0)
    selector.offer(candidate(1, 0.2))
    selector.offer(candidate(2, 0.3))
    selector.offer(candidate(3, 0.9))
    assert [item.frame_index for item in selector.get("A", 1)] == [3, 2]


def test_nearby_better_candidate_replaces_weaker():
    selector = PlateBestFrameSelector(top_k=3, min_frame_gap=2)
    selector.offer(candidate(10, 0.4))
    assert selector.offer(candidate(11, 0.8))
    assert [item.frame_index for item in selector.get("A", 1)] == [11]


def test_deterministic_ties_use_confidence_then_earlier_frame():
    selector = PlateBestFrameSelector(top_k=3, min_frame_gap=0)
    for item in (candidate(3, 0.8, 0.8), candidate(2, 0.8, 0.9), candidate(1, 0.8, 0.9)):
        selector.offer(item)
    assert [item.frame_index for item in selector.get("A", 1)] == [1, 2, 3]


def test_camera_and_track_state_are_isolated():
    selector = PlateBestFrameSelector()
    selector.offer(candidate(1, 0.5, camera="A", track=1))
    selector.offer(candidate(1, 0.6, camera="B", track=1))
    selector.offer(candidate(1, 0.7, camera="A", track=2))
    assert selector.get("A", 1)[0].quality_score == 0.5
    assert selector.get("B", 1)[0].quality_score == 0.6
    assert selector.get("A", 2)[0].quality_score == 0.7


def test_finalize_returns_candidates_and_releases_state():
    selector = PlateBestFrameSelector()
    selector.offer(candidate(1, 0.5))
    assert len(selector.finalize("A", 1)) == 1
    assert selector.get("A", 1) == ()


def test_fallback_fills_missing_slots_after_primaries():
    selector = PlateBestFrameSelector(top_k=3, min_frame_gap=0)
    selector.offer(candidate(1, .4, tier="PRIMARY"))
    selector.offer(candidate(2, .9, tier="FALLBACK"))
    selector.offer(candidate(3, .6, tier="PRIMARY"))
    assert [(item.selection_tier, item.frame_index) for item in selector.get("A", 1)] == [
        ("PRIMARY", 3), ("PRIMARY", 1), ("FALLBACK", 2),
    ]


def test_fallback_cannot_displace_primary_when_primary_slots_are_full():
    selector = PlateBestFrameSelector(top_k=3, min_frame_gap=0)
    for frame, score in ((1, .3), (2, .4), (3, .5)):
        selector.offer(candidate(frame, score, tier="PRIMARY"))
    assert not selector.offer(candidate(4, 1.0, tier="FALLBACK"))
    assert all(item.selection_tier == "PRIMARY" for item in selector.get("A", 1))


def test_later_primary_displaces_selected_fallback():
    selector = PlateBestFrameSelector(top_k=2, min_frame_gap=0)
    selector.offer(candidate(1, .9, tier="FALLBACK"))
    selector.offer(candidate(2, .2, tier="PRIMARY"))
    selector.offer(candidate(3, .1, tier="PRIMARY"))
    assert [item.frame_index for item in selector.get("A", 1)] == [2, 3]


def test_primary_wins_min_frame_gap_against_fallback():
    selector = PlateBestFrameSelector(top_k=3, min_frame_gap=2)
    selector.offer(candidate(10, .9, tier="FALLBACK"))
    selector.offer(candidate(11, .1, tier="PRIMARY"))
    assert [(item.selection_tier, item.frame_index) for item in selector.get("A", 1)] == [
        ("PRIMARY", 11),
    ]


def test_tier_ordering_is_deterministic():
    values = [
        candidate(4, .7, .8, tier="FALLBACK"),
        candidate(3, .7, .8, tier="FALLBACK"),
        candidate(2, .5, .8, tier="PRIMARY"),
        candidate(1, .5, .8, tier="PRIMARY"),
    ]
    first = PlateBestFrameSelector(top_k=4, min_frame_gap=0)
    second = PlateBestFrameSelector(top_k=4, min_frame_gap=0)
    for item in values:
        first.offer(item)
    for item in reversed(values):
        second.offer(item)
    assert [item.frame_index for item in first.get("A", 1)] == [1, 2, 3, 4]
    assert first.get("A", 1) == second.get("A", 1)
