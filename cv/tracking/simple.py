from collections.abc import Sequence

from cv.common.frame import FramePacket
from cv.detection.models import VehicleDetection
from cv.tracking.association import greedy_iou_matching
from cv.tracking.models import TrackState, TrackStatus
from cv.tracking.tracker import VehicleTracker


class SimpleVehicleTracker(VehicleTracker):
    """Deterministic IoU-based tracker for pipeline development."""

    supports_lifecycle_events = True

    def __init__(
        self,
        minimum_iou: float = 0.3,
        max_missed_frames: int = 30,
    ) -> None:
        if not 0.0 <= minimum_iou <= 1.0:
            raise ValueError("minimum_iou must be between 0 and 1")

        if max_missed_frames < 0:
            raise ValueError("max_missed_frames must be non-negative")

        self.minimum_iou = minimum_iou
        self.max_missed_frames = max_missed_frames

        self._next_track_id = 1
        self._tracks: dict[int, TrackState] = {}
        self._completed_tracks: dict[int, TrackState] = {}

    def update(
        self,
        detections: Sequence[VehicleDetection],
    ) -> Sequence[TrackState]:
        """Associate detections with existing tracks."""

        active_tracks = [
            track
            for track in self._tracks.values()
            if track.missed_frames <= self.max_missed_frames
        ]

        track_inputs = [
            (track.track_id, track.bbox)
            for track in active_tracks
        ]

        detection_boxes = [
            detection.bbox
            for detection in detections
        ]

        matches, unmatched_track_ids, unmatched_detection_indices = (
            greedy_iou_matching(
                track_inputs,
                detection_boxes,
                minimum_iou=self.minimum_iou,
            )
        )

        tracks_by_id = {
            track.track_id: track
            for track in active_tracks
        }

        # Update tracks that matched a detection.
        for track_id, detection_index in matches:
            tracks_by_id[track_id].update(
                detections[detection_index]
            )

        # Mark unmatched tracks as lost.
        for track_id in unmatched_track_ids:
            track = tracks_by_id[track_id]
            track.missed_frames += 1
            track.age += 1
            track.status = TrackStatus.LOST

        # Create tracks for genuinely new detections.
        for detection_index in unmatched_detection_indices:
            detection = detections[detection_index]

            track = TrackState(
                track_id=self._next_track_id,
                camera_id=detection.frame.camera_id,
                bbox=detection.bbox,
                class_name=detection.class_name,
                first_frame_index=detection.frame.frame_index,
                last_frame_index=detection.frame.frame_index,
                first_timestamp=detection.frame.timestamp,
                last_timestamp=detection.frame.timestamp,
                detection_confidence=detection.confidence,
                history=[detection.bbox],
                status=TrackStatus.ACTIVE,
            )

            self._tracks[self._next_track_id] = track
            self._next_track_id += 1

        # Archive tracks that have exceeded the allowed gap.
        expired_track_ids = [
            track_id
            for track_id, track in self._tracks.items()
            if track.missed_frames > self.max_missed_frames
        ]

        for track_id in expired_track_ids:
            track = self._tracks[track_id]
            track.status = TrackStatus.EXPIRED
            self._completed_tracks[track_id] = track
            del self._tracks[track_id]

        # Return only tracks that currently have a detection.
        return [
            track
            for track in self._tracks.values()
            if track.missed_frames == 0
        ]

    def pop_expired_tracks(self) -> Sequence[TrackState]:
        """Drain permanent-retirement events in deterministic ID order."""

        tracks = tuple(
            self._completed_tracks[track_id]
            for track_id in sorted(self._completed_tracks)
        )
        self._completed_tracks.clear()
        return tracks

    def advance_without_detection(self, frame: FramePacket) -> Sequence[TrackState]:
        # Detector cadence gaps are not negative observations. Preserve bbox,
        # missed-frame counters, and lifecycle expiry while advancing age.
        active = []
        for track in self._tracks.values():
            track.age += 1
            if track.missed_frames == 0:
                active.append(track)
        return tuple(sorted(active, key=lambda item: item.track_id))

    def finalize_all(self) -> Sequence[TrackState]:
        """Drain pending expiry events and all active/lost tracks."""

        tracks = list(self.pop_expired_tracks())
        tracks.extend(self._tracks.values())
        self._tracks.clear()
        return tuple(sorted(
            tracks,
            key=lambda track: (track.camera_id, track.track_id),
        ))
