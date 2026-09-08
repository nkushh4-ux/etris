from collections.abc import Sequence
from datetime import datetime

from cv.common.frame import FramePacket
from cv.detection.detector import VehicleDetector
from cv.detection.models import VehicleDetection
from cv.ocr.coordinator import TrackOCRCoordinator
from cv.pipeline.models import CompletedTrackANPR, TrackTerminationReason
from cv.pipeline.timing import StageTimingProfiler
from cv.plate.detector import PlateDetector
from cv.plate.models import PlateCandidate
from cv.plate.quality import PlateQualityAssessor
from cv.plate.selection import PlateBestFrameSelector
from cv.tracking.models import TrackState
from cv.tracking.tracker import VehicleTracker


class CameraPipeline:
    """Process frames from one camera through detection and tracking."""

    def __init__(
        self,
        detector: VehicleDetector,
        tracker: VehicleTracker,
        plate_detector: PlateDetector | None = None,
        quality_assessor: PlateQualityAssessor | None = None,
        best_frame_selector: PlateBestFrameSelector | None = None,
        ocr_coordinator: TrackOCRCoordinator | None = None,
        vehicle_detection_interval: int = 1,
        plate_detection_intervals: tuple[int, int, int, int] = (1, 1, 1, 1),
        profiler: StageTimingProfiler | None = None,
    ) -> None:
        self.detector = detector
        self.tracker = tracker
        self.plate_detector = plate_detector
        self.quality_assessor = quality_assessor
        self.best_frame_selector = best_frame_selector
        self.ocr_coordinator = ocr_coordinator
        if vehicle_detection_interval < 1 or any(value < 1 for value in plate_detection_intervals):
            raise ValueError("performance intervals must be positive")
        self.vehicle_detection_interval = vehicle_detection_interval
        self.plate_detection_intervals = plate_detection_intervals
        self.profiler = profiler or StageTimingProfiler()
        self._last_plate_attempt: dict[tuple[str, int], int] = {}
        self._completed_tracks: list[CompletedTrackANPR] = []
        self._last_frame: FramePacket | None = None

        stage_three = (
            plate_detector,
            quality_assessor,
            best_frame_selector,
        )
        if any(stage_three) and not all(stage_three):
            raise ValueError("Stage 3 components must be supplied together")
        if ocr_coordinator is not None and best_frame_selector is None:
            raise ValueError("OCR coordination requires Stage 3 components")
        if (
            ocr_coordinator is not None
            and not tracker.supports_lifecycle_events
        ):
            raise ValueError(
                "OCR lifecycle integration requires a tracker that supports "
                "lifecycle events (pop_expired_tracks and finalize_all)"
            )

    def process_frame(
        self,
        frame: FramePacket,
    ) -> Sequence[TrackState]:
        """Detect and track vehicles in one frame."""

        if frame.frame_index % self.vehicle_detection_interval == 0:
            with self.profiler.measure("vehicle_detection"):
                detections: Sequence[VehicleDetection] = self.detector.detect(frame)
            with self.profiler.measure("tracker_update"):
                tracks = self.tracker.update(detections)
        else:
            with self.profiler.measure("tracker_update"):
                tracks = self.tracker.advance_without_detection(frame)
        self._last_frame = frame

        for expired_track in self.tracker.pop_expired_tracks():
            self._complete_track(
                expired_track,
                TrackTerminationReason.TRACK_EXPIRED,
                frame.frame_index,
                frame.timestamp,
            )

        if self.plate_detector is not None:
            eligible_tracks = []
            for track in tracks:
                key = (track.camera_id, track.track_id)
                retained = self.best_frame_selector.get(*key)
                interval = self.plate_detection_intervals[min(len(retained), 3)]
                previous = self._last_plate_attempt.get(key)
                if previous is not None and frame.frame_index - previous < interval:
                    continue
                self._last_plate_attempt[key] = frame.frame_index
                eligible_tracks.append(track)
            if eligible_tracks:
                with self.profiler.measure("plate_detection"):
                    if hasattr(self.plate_detector, "detect_batch_in_frame"):
                        plate_detections = self.plate_detector.detect_batch_in_frame(
                            eligible_tracks, frame.image, frame
                        )
                    else:
                        plate_detections = tuple(
                            detection for track in eligible_tracks
                            for detection in self.plate_detector.detect_in_frame(track, frame.image, frame)
                        )
                for plate_detection in plate_detections:
                    with self.profiler.measure("stage3_quality_selection"):
                        candidate = self.quality_assessor.assess(plate_detection)
                        if candidate is not None:
                            self.best_frame_selector.offer(candidate)

        return tracks

    def _complete_track(
        self,
        track: TrackState,
        reason: TrackTerminationReason,
        frame_index: int,
        timestamp: datetime,
    ) -> None:
        """Finalize Stage 3, then OCR, then fusion for one retired track."""

        errors: list[str] = []
        try:
            candidates = (
                self.best_frame_selector.finalize(track.camera_id, track.track_id)
                if self.best_frame_selector is not None else ()
            )
        except Exception as exc:
            candidates = ()
            errors.append(f"stage3_finalization_error:{type(exc).__name__}")

        if self.ocr_coordinator is None:
            # Legacy pipelines release Stage 3 state but have no ANPR output.
            return

        ocr_results = []
        for candidate in candidates:
            try:
                with self.profiler.measure("canonical_ocr"):
                    ocr_results.append(self.ocr_coordinator.add_candidate(candidate))
            except Exception as exc:
                errors.append(f"ocr_coordination_error:{type(exc).__name__}")

        # Always drain fusion state, including the all-failure/no-plate case.
        with self.profiler.measure("fusion_lifecycle"):
            fused = self.ocr_coordinator.finalize_track(track.camera_id, track.track_id)
        self._completed_tracks.append(CompletedTrackANPR(
            camera_id=track.camera_id,
            track_id=track.track_id,
            termination_reason=reason,
            termination_frame_index=frame_index,
            timestamp=timestamp,
            selected_candidate_count=len(candidates),
            ocr_result_count=len(ocr_results),
            ocr_results=tuple(ocr_results),
            fused_plate_result=fused,
            finalization_errors=tuple(errors),
        ))

    def pop_completed_tracks(self) -> tuple[CompletedTrackANPR, ...]:
        """Drain terminal results so the pipeline does not retain them."""

        completed = tuple(self._completed_tracks)
        self._completed_tracks.clear()
        return completed

    def finalize(self) -> tuple[CompletedTrackANPR, ...]:
        """Idempotently finalize all tracks remaining at end-of-stream."""

        pending_expired = self.tracker.pop_expired_tracks()
        remaining = self.tracker.finalize_all()
        groups = (
            (pending_expired, TrackTerminationReason.TRACK_EXPIRED),
            (remaining, TrackTerminationReason.END_OF_STREAM),
        )
        for tracks, reason in groups:
            for track in sorted(
                tracks, key=lambda item: (item.camera_id, item.track_id)
            ):
                if self._last_frame is None:
                    frame_index = track.last_frame_index
                    timestamp = track.last_timestamp
                else:
                    frame_index = self._last_frame.frame_index
                    timestamp = self._last_frame.timestamp
                self._complete_track(
                    track,
                    reason,
                    frame_index,
                    timestamp,
                )
        return self.pop_completed_tracks()

    def finalize_track(
        self,
        camera_id: str,
        track_id: int,
    ) -> tuple[PlateCandidate, ...]:
        """Finalize and release retained Stage 3 crops for one track."""

        if self.best_frame_selector is None:
            return ()
        return self.best_frame_selector.finalize(camera_id, track_id)
