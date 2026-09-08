from cv.detection.models import BoundingBox
from cv.tracking.association import (
    greedy_iou_matching,
    intersection_over_union,
)


def test_iou_identical_boxes():
    box = BoundingBox(0, 0, 100, 100)

    assert intersection_over_union(box, box) == 1.0


def test_iou_non_overlapping_boxes():
    first = BoundingBox(0, 0, 10, 10)
    second = BoundingBox(20, 20, 30, 30)

    assert intersection_over_union(first, second) == 0.0


def test_iou_partial_overlap():
    first = BoundingBox(0, 0, 100, 100)
    second = BoundingBox(50, 50, 150, 150)

    expected = 2500 / 17500

    assert intersection_over_union(first, second) == expected


def test_greedy_matching():
    tracks = [
        (1, BoundingBox(0, 0, 100, 100)),
        (2, BoundingBox(200, 200, 300, 300)),
    ]

    detections = [
        BoundingBox(5, 5, 105, 105),
        BoundingBox(205, 205, 305, 305),
    ]

    matches, unmatched_tracks, unmatched_detections = (
        greedy_iou_matching(
            tracks,
            detections,
            minimum_iou=0.3,
        )
    )

    assert matches == [(1, 0), (2, 1)]
    assert unmatched_tracks == []
    assert unmatched_detections == []


def test_greedy_matching_reports_unmatched():
    tracks = [
        (1, BoundingBox(0, 0, 100, 100)),
    ]

    detections = [
        BoundingBox(500, 500, 600, 600),
    ]

    matches, unmatched_tracks, unmatched_detections = (
        greedy_iou_matching(
            tracks,
            detections,
            minimum_iou=0.3,
        )
    )

    assert matches == []
    assert unmatched_tracks == [1]
    assert unmatched_detections == [0]


def test_one_detection_matches_at_most_one_of_two_tracks():
    tracks = [
        (1, BoundingBox(0, 0, 100, 100)),
        (2, BoundingBox(10, 0, 110, 100)),
    ]
    detection = BoundingBox(5, 0, 105, 100)

    matches, unmatched_tracks, unmatched_detections = greedy_iou_matching(
        tracks, [detection], minimum_iou=0.3
    )

    assert matches == [(1, 0)]
    assert unmatched_tracks == [2]
    assert unmatched_detections == []
    assert len({detection_index for _, detection_index in matches}) == len(matches)


def test_exact_duplicate_detection_boxes_are_one_observation():
    box = BoundingBox(0, 0, 100, 100)
    matches, unmatched_tracks, unmatched_detections = greedy_iou_matching(
        [(1, box)], [box, box], minimum_iou=0.3
    )

    assert matches == [(1, 0)]
    assert unmatched_tracks == []
    assert unmatched_detections == []


def test_matching_is_independent_of_track_input_order():
    tracks = [
        (1, BoundingBox(0, 0, 100, 100)),
        (2, BoundingBox(10, 0, 110, 100)),
    ]
    detections = [BoundingBox(5, 0, 105, 100)]

    forward = greedy_iou_matching(tracks, detections, minimum_iou=0.3)
    reverse = greedy_iou_matching(list(reversed(tracks)), detections, minimum_iou=0.3)

    assert forward[0] == reverse[0] == [(1, 0)]
