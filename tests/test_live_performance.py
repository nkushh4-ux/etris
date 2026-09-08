import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from backend.streaming.preview import (
    AsyncPreviewingBestFrameSelector,
    LivePreviewManager,
)
from cv.common.frame import FramePacket
from cv.detection.models import BoundingBox, VehicleDetection
from cv.ocr.models import OCRResultStatus
from cv.pipeline.camera import CameraPipeline
from cv.plate.ultralytics_plate_detector import UltralyticsPlateDetector
from cv.tracking.simple import SimpleVehicleTracker

BASE = datetime(2026, 9, 5, tzinfo=UTC)


def frame(index):
    import numpy as np
    return FramePacket("CAM", index, BASE + timedelta(milliseconds=33 * index),
                       np.zeros((100, 160, 3), dtype="uint8"))


class Detector:
    def __init__(self): self.frames = []
    def detect(self, packet):
        self.frames.append((packet.frame_index, packet.timestamp))
        return [VehicleDetection(packet, BoundingBox(10, 10, 90, 80), "car", .9, "fake", "1")]


def test_vehicle_cadence_is_deterministic_and_source_frames_remain_continuous():
    detector = Detector()
    pipeline = CameraPipeline(detector, SimpleVehicleTracker(), vehicle_detection_interval=2)
    returned = [pipeline.process_frame(frame(index))[0].track_id for index in range(6)]
    assert detector.frames == [(0, BASE), (2, BASE + timedelta(milliseconds=66)),
                               (4, BASE + timedelta(milliseconds=132))]
    assert returned == [1] * 6
    assert pipeline.tracker._tracks[1].missed_frames == 0


def test_detector_miss_lifecycle_is_unchanged_after_cadence_bridge():
    tracker = SimpleVehicleTracker(max_missed_frames=1)
    detection = Detector().detect(frame(0))
    tracker.update(detection)
    tracker.advance_without_detection(frame(1))
    assert tracker.pop_expired_tracks() == ()
    tracker.update(()); assert tracker.pop_expired_tracks() == ()
    tracker.update(()); assert len(tracker.pop_expired_tracks()) == 1


def test_plate_scheduler_refreshes_and_never_permanently_starves_track():
    class Plate:
        def __init__(self): self.frames = []
        def detect_in_frame(self, track, image, packet): self.frames.append(packet.frame_index); return ()
    class Selector:
        def get(self, camera, track): return (1, 2, 3)
        def offer(self, candidate): return True
        def finalize(self, camera, track): return ()
    plate = Plate()
    pipeline = CameraPipeline(Detector(), SimpleVehicleTracker(), plate, SimpleNamespace(), Selector(),
                              plate_detection_intervals=(2, 3, 4, 8))
    for index in range(18): pipeline.process_frame(frame(index))
    assert plate.frames == [0, 8, 16]


def test_async_preview_queue_is_bounded_and_remains_presentation_only():
    gate = threading.Event()
    class Selector:
        def offer(self, item): return True
        def get(self, camera, track): return ()
        def finalize(self, camera, track): return ("canonical",)
    class Engine:
        def recognize(self, item):
            gate.wait(.2)
            return SimpleNamespace(status=OCRResultStatus.SUCCESS, raw_text="X",
                                   normalized_text="X", confidence=.8)
    wrapper = AsyncPreviewingBestFrameSelector(Selector(), Engine(), LivePreviewManager(1),
                                               lambda *_: None, 30, queue_size=2)
    for index in range(20):
        wrapper.offer(SimpleNamespace(camera_id="CAM", track_id=1, frame_index=index,
                                      quality_score=.8, timestamp=BASE))
    assert wrapper.pending_jobs <= 2
    assert wrapper.finalize("CAM", 1) == ("canonical",)
    gate.set(); wrapper.close()


def test_timing_report_contains_distribution_fields():
    pipeline = CameraPipeline(Detector(), SimpleVehicleTracker())
    pipeline.process_frame(frame(0))
    timing = pipeline.profiler.report()["vehicle_detection"]
    assert {"count", "mean_ms", "median_ms", "p95_ms", "percent"} <= timing.keys()


def test_plate_detector_batches_rois_and_maps_each_result_to_its_track():
    import numpy as np

    class Value:
        def __init__(self, value): self.value = value
        def item(self): return self.value
    class Box:
        def __init__(self, coords): self.xyxy = np.asarray([coords]); self.conf = Value(.8)
    class Model:
        def __init__(self): self.batch_sizes = []
        def predict(self, **kwargs):
            self.batch_sizes.append(len(kwargs["source"]))
            return [SimpleNamespace(boxes=[Box((1, 2, 11, 7))]) for _ in kwargs["source"]]

    detector = object.__new__(UltralyticsPlateDetector)
    detector.confidence_threshold = .25; detector.device = "cpu"; detector.input_size = 640
    detector.roi_padding = 0.0; detector.half_precision = False; detector.max_batch_size = 2
    detector.weights = Path("plate.pt"); detector.model = Model()
    detector.model_invocations = detector.rois_processed = 0
    tracks = [SimpleNamespace(track_id=index + 1, bbox=BoundingBox(index * 30, 10, index * 30 + 20, 30))
              for index in range(3)]
    packet = frame(0)
    detections = detector.detect_batch_in_frame(tracks, packet.image, packet)
    assert detector.model.batch_sizes == [2, 1]
    assert detector.model_invocations == 2 and detector.rois_processed == 3
    assert [item.track_id for item in detections] == [1, 2, 3]
    assert [item.bbox.x1 for item in detections] == [1.0, 31.0, 61.0]
