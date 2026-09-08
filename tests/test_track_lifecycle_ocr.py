from collections.abc import Sequence
from datetime import UTC, datetime

import numpy as np
import pytest

from cv.common.frame import FramePacket
from cv.detection.detector import VehicleDetector
from cv.detection.models import BoundingBox, VehicleDetection
from cv.ocr.coordinator import TrackOCRCoordinator
from cv.ocr.engine import OCREngine
from cv.ocr.fusion import MultiFrameOCRFuser
from cv.ocr.models import FusionStatus, OCRResult, OCRResultStatus
from cv.ocr.normalization import normalize_plate_text
from cv.pipeline.camera import CameraPipeline
from cv.pipeline.models import TrackTerminationReason
from cv.plate.models import PlateDetection
from cv.plate.quality import PlateQualityAssessor, PlateQualityConfig
from cv.plate.selection import PlateBestFrameSelector
from cv.tracking.simple import SimpleVehicleTracker
from cv.tracking.tracker import VehicleTracker


def frame(index, camera="A"):
    return FramePacket(camera, index, datetime.now(UTC),
                       np.full((100, 180, 3), 120, dtype=np.uint8))


class ScheduledDetector(VehicleDetector):
    def __init__(self, schedule):
        self.schedule = schedule

    def detect(self, packet: FramePacket) -> Sequence[VehicleDetection]:
        boxes = self.schedule.get(packet.frame_index, ())
        return [VehicleDetection(packet, box, "car", .95, "fake", "1")
                for box in boxes]


class PlateDetector:
    def __init__(self, skip=()):
        self.skip = set(skip)

    def detect_in_frame(self, track, image, packet):
        if packet.frame_index in self.skip:
            return ()
        return (PlateDetection(packet, track.track_id, BoundingBox(20, 30, 120, 55),
                               .9, "plate", "1"),)


class FakeEngine(OCREngine):
    def __init__(self, texts=None, failed=()):
        self.texts = texts or {}
        self.failed = set(failed)
        self.frames = []

    def recognize(self, candidate):
        self.frames.append(candidate.frame_index)
        failed = candidate.frame_index in self.failed
        raw = "" if failed else self.texts.get(candidate.frame_index, "KA02MM9091")
        return OCRResult(
            candidate.camera_id, candidate.track_id, candidate.frame_index,
            raw, normalize_plate_text(raw), 0.0 if failed else .95,
            "fake", "1", f"{candidate.camera_id}-{candidate.track_id}-{candidate.frame_index}",
            candidate.quality_score, candidate.detector_confidence,
            (candidate.bbox.x1, candidate.bbox.y1, candidate.bbox.x2, candidate.bbox.y2),
            candidate.timestamp,
            OCRResultStatus.FAILED if failed else OCRResultStatus.SUCCESS,
            "inference_error" if failed else None,
        )


def pipeline(schedule, *, camera="A", max_missed=1, engine=None, skip=()):
    selector = PlateBestFrameSelector(top_k=3, min_frame_gap=0)
    engine = engine or FakeEngine()
    coordinator = TrackOCRCoordinator(engine, MultiFrameOCRFuser())
    pipe = CameraPipeline(
        ScheduledDetector(schedule),
        SimpleVehicleTracker(minimum_iou=.3, max_missed_frames=max_missed),
        PlateDetector(skip),
        PlateQualityAssessor(PlateQualityConfig(
            min_plate_width_px=4, min_plate_height_px=4,
        )),
        selector,
        coordinator,
    )
    return pipe, selector, coordinator, engine


BOX = BoundingBox(10, 10, 150, 80)


class NonLifecycleTracker(VehicleTracker):
    def update(self, detections):
        return ()


def test_simple_tracker_reports_lifecycle_support():
    assert SimpleVehicleTracker().supports_lifecycle_events is True


def test_base_tracker_reports_no_lifecycle_support():
    assert NonLifecycleTracker().supports_lifecycle_events is False


def test_ocr_pipeline_rejects_tracker_without_lifecycle_support():
    selector = PlateBestFrameSelector()
    coordinator = TrackOCRCoordinator(FakeEngine(), MultiFrameOCRFuser())
    with pytest.raises(ValueError, match="requires a tracker that supports lifecycle"):
        CameraPipeline(
            ScheduledDetector({}),
            NonLifecycleTracker(),
            PlateDetector(),
            PlateQualityAssessor(PlateQualityConfig()),
            selector,
            coordinator,
        )


def test_non_ocr_pipeline_accepts_tracker_without_lifecycle_support():
    pipe = CameraPipeline(ScheduledDetector({}), NonLifecycleTracker())
    assert pipe.process_frame(frame(0)) == ()


def test_supported_tracker_lifecycle_behavior_remains_unchanged():
    pipe, _, _, _ = pipeline({0: (BOX,)}, max_missed=0)
    pipe.process_frame(frame(0))
    pipe.process_frame(frame(1))
    completed = pipe.pop_completed_tracks()
    assert len(completed) == 1
    assert completed[0].termination_reason is TrackTerminationReason.TRACK_EXPIRED


def test_active_and_one_missed_frame_do_not_finalize():
    pipe, _, _, _ = pipeline({0: (BOX,)}, max_missed=1)
    pipe.process_frame(frame(0)); pipe.process_frame(frame(1))
    assert pipe.pop_completed_tracks() == ()


def test_reacquired_track_keeps_identity_without_expiration():
    pipe, _, _, _ = pipeline({0: (BOX,), 2: (BoundingBox(12, 10, 152, 80),)},
                              max_missed=2)
    assert pipe.process_frame(frame(0))[0].track_id == 1
    pipe.process_frame(frame(1))
    assert pipe.process_frame(frame(2))[0].track_id == 1
    assert pipe.pop_completed_tracks() == ()


def test_true_expiry_emits_once_and_releases_all_state():
    pipe, selector, coordinator, engine = pipeline({0: (BOX,), 1: (BOX,)},
                                                    max_missed=1)
    for index in range(4):
        pipe.process_frame(frame(index))
    completed = pipe.pop_completed_tracks()
    assert len(completed) == 1
    item = completed[0]
    assert item.termination_reason is TrackTerminationReason.TRACK_EXPIRED
    assert item.selected_candidate_count == 2
    assert item.ocr_result_count == 2 and engine.frames == [0, 1]
    assert item.fused_plate_result.status is FusionStatus.ACCEPTED
    assert selector.get("A", 1) == ()
    assert ("A", 1) not in coordinator.fuser._tracks
    assert pipe.pop_completed_tracks() == ()
    pipe.process_frame(frame(4))
    assert pipe.pop_completed_tracks() == ()


def test_low_confidence_and_unknown_terminal_results_are_preserved():
    weak = FakeEngine(texts={0: "KA02MM9091", 1: "KA02MM9091"})
    original = weak.recognize
    def recognize(candidate):
        result = original(candidate)
        return OCRResult(*(
            result.camera_id, result.track_id, result.frame_index, result.raw_text,
            result.normalized_text, .01, result.engine, result.engine_version,
            result.evidence_id, .01, result.detector_confidence, result.bbox,
            result.timestamp, result.status, result.rejected_reason,
        ))
    weak.recognize = recognize
    low, _, _, _ = pipeline({0: (BOX,), 1: (BOX,)}, engine=weak)
    for index in range(4): low.process_frame(frame(index))
    assert low.pop_completed_tracks()[0].fused_plate_result.status is FusionStatus.LOW_CONFIDENCE

    empty, _, _, _ = pipeline({0: (BOX,)}, skip=(0,))
    for index in range(3): empty.process_frame(frame(index))
    result = empty.pop_completed_tracks()[0]
    assert result.selected_candidate_count == 0
    assert result.fused_plate_result.status is FusionStatus.UNKNOWN


def test_one_ocr_failure_continues_and_all_failures_are_unknown():
    partial = FakeEngine(failed=(1,))
    pipe, _, _, _ = pipeline({0: (BOX,), 1: (BOX,), 2: (BOX,)}, engine=partial)
    for index in range(5): pipe.process_frame(frame(index))
    item = pipe.pop_completed_tracks()[0]
    assert len(item.ocr_results) == 3
    assert item.ocr_results[1].status is OCRResultStatus.FAILED
    assert item.fused_plate_result.status is FusionStatus.ACCEPTED

    failed = FakeEngine(failed=(0, 1))
    pipe, _, _, _ = pipeline({0: (BOX,), 1: (BOX,)}, engine=failed)
    for index in range(4): pipe.process_frame(frame(index))
    assert pipe.pop_completed_tracks()[0].fused_plate_result.status is FusionStatus.UNKNOWN


def test_multiple_tracks_expire_in_deterministic_order():
    left, right = BoundingBox(5, 5, 70, 70), BoundingBox(100, 5, 170, 70)
    pipe, _, _, _ = pipeline({0: (right, left)})
    for index in range(3): pipe.process_frame(frame(index))
    completed = pipe.pop_completed_tracks()
    assert [item.track_id for item in completed] == [1, 2]
    assert all(item.termination_reason is TrackTerminationReason.TRACK_EXPIRED
               for item in completed)


def test_end_of_stream_flushes_active_tracks_once_and_cleans_state():
    pipe, selector, coordinator, _ = pipeline({0: (BOX,), 1: (BOX,)}, max_missed=5)
    pipe.process_frame(frame(0)); pipe.process_frame(frame(1))
    completed = pipe.finalize()
    assert len(completed) == 1
    assert completed[0].termination_reason is TrackTerminationReason.END_OF_STREAM
    assert completed[0].fused_plate_result.status is FusionStatus.ACCEPTED
    assert pipe.finalize() == ()
    assert selector.get("A", 1) == ()
    assert coordinator.fuser._tracks == {}
    assert pipe.tracker._tracks == {}


def test_same_local_track_id_isolated_across_camera_pipelines():
    shared_fuser = MultiFrameOCRFuser()
    outputs = []
    for camera in ("A", "B"):
        selector = PlateBestFrameSelector(top_k=3, min_frame_gap=0)
        coordinator = TrackOCRCoordinator(FakeEngine(), shared_fuser)
        pipe = CameraPipeline(ScheduledDetector({0: (BOX,), 1: (BOX,)}),
                              SimpleVehicleTracker(max_missed_frames=1),
                              PlateDetector(), PlateQualityAssessor(PlateQualityConfig(
                                  min_plate_width_px=4, min_plate_height_px=4)),
                              selector, coordinator)
        pipe.process_frame(frame(0, camera)); pipe.process_frame(frame(1, camera))
        outputs.extend(pipe.finalize())
    assert [(item.camera_id, item.track_id) for item in outputs] == [("A", 1), ("B", 1)]
    assert shared_fuser._tracks == {}


def test_tracker_expiration_event_is_consumed_exactly_once():
    tracker = SimpleVehicleTracker(max_missed_frames=0)
    detection = ScheduledDetector({0: (BOX,)}).detect(frame(0))[0]
    tracker.update((detection,)); tracker.update(())
    assert [track.track_id for track in tracker.pop_expired_tracks()] == [1]
    assert tracker.pop_expired_tracks() == ()
    assert tracker._tracks == {} and tracker._completed_tracks == {}
