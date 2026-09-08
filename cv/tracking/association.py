from collections.abc import Sequence

from cv.detection.models import BoundingBox


def intersection_over_union(
    first: BoundingBox,
    second: BoundingBox,
) -> float:
    """Calculate IoU between two bounding boxes."""

    intersection_x1 = max(first.x1, second.x1)
    intersection_y1 = max(first.y1, second.y1)
    intersection_x2 = min(first.x2, second.x2)
    intersection_y2 = min(first.y2, second.y2)

    intersection_width = max(
        0.0,
        intersection_x2 - intersection_x1,
    )
    intersection_height = max(
        0.0,
        intersection_y2 - intersection_y1,
    )

    intersection_area = (
        intersection_width * intersection_height
    )

    first_area = first.width * first.height
    second_area = second.width * second.height

    union_area = first_area + second_area - intersection_area

    if union_area <= 0.0:
        return 0.0

    return intersection_area / union_area


def greedy_iou_matching(
    tracks: Sequence[tuple[int, BoundingBox]],
    detections: Sequence[BoundingBox],
    minimum_iou: float = 0.3,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Match detections to tracks using greedy IoU association.

    Returns:
        matches:
            Pairs of (track_id, detection_index).

        unmatched_tracks:
            Track IDs that received no detection.

        unmatched_detections:
            Detection indices that matched no track.
    """

    # Exact duplicate detector boxes describe the same geometric observation.
    # Keep the first index as canonical so a duplicated model output cannot
    # update/create multiple tracks in the same frame.
    unique_detection_indices: list[int] = []
    seen_detection_boxes: set[tuple[float, float, float, float]] = set()
    for detection_index, detection_bbox in enumerate(detections):
        key = (
            detection_bbox.x1,
            detection_bbox.y1,
            detection_bbox.x2,
            detection_bbox.y2,
        )
        if key not in seen_detection_boxes:
            seen_detection_boxes.add(key)
            unique_detection_indices.append(detection_index)

    candidates: list[tuple[float, int, int]] = []

    for track_id, track_bbox in tracks:
        for detection_index in unique_detection_indices:
            detection_bbox = detections[detection_index]
            iou = intersection_over_union(
                track_bbox,
                detection_bbox,
            )

            if iou >= minimum_iou:
                candidates.append(
                    (iou, track_id, detection_index)
                )

    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    used_tracks: set[int] = set()
    used_detections: set[int] = set()

    matches: list[tuple[int, int]] = []

    for _, track_id, detection_index in candidates:
        if track_id in used_tracks:
            continue

        if detection_index in used_detections:
            continue

        matches.append(
            (track_id, detection_index)
        )

        used_tracks.add(track_id)
        used_detections.add(detection_index)

    unmatched_tracks = [
        track_id
        for track_id, _ in tracks
        if track_id not in used_tracks
    ]

    unmatched_detections = [
        detection_index
        for detection_index in unique_detection_indices
        if detection_index not in used_detections
    ]

    return (
        matches,
        unmatched_tracks,
        unmatched_detections,
    )
