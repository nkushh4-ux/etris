from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import numpy as np
import pytest

from cv.detection.models import BoundingBox
from cv.ocr.coordinator import TrackOCRCoordinator
from cv.ocr.engine import result_to_evidence
from cv.ocr.fusion import MultiFrameOCRFuser
from cv.ocr.models import FusionStatus, OCRResultStatus
from cv.ocr.svtrv2 import SVTRv2Config, SVTRv2Engine
from cv.plate.models import PlateCandidate, PlateQualityMetrics


def candidate(frame=1, camera="A", track=1, crop=None):
    metrics = PlateQualityMetrics(0.9, 0.8, 0.7, 0.9, 0.95, 1.0,
                                  200.0, 128.0, 50.0, 120, 32, 1.0)
    return PlateCandidate(camera, track, frame, datetime.now(UTC),
                          BoundingBox(1, 2, 121, 34), 0.95, metrics, 0.85,
                          np.ones((32, 120, 3), dtype=np.uint8) if crop is None else crop)


class Backend:
    def __init__(self, outputs=None, error=None):
        self.outputs = outputs or [{"text": " ka-02 mm 9091 ", "score": 0.93}]
        self.error = error
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        assert "img_numpy" in kwargs
        if self.error:
            raise self.error
        return self.outputs


@pytest.fixture
def config(tmp_path):
    root = tmp_path / "OpenOCR"
    root.mkdir()
    cfg = tmp_path / "config.yml"; cfg.touch()
    checkpoint = tmp_path / "best.pth"; checkpoint.write_bytes(b"model")
    return SVTRv2Config(cfg, checkpoint, root)


def make_engine(config, backend, loads=None):
    def factory(_):
        if loads is not None: loads.append(1)
        return backend
    return SVTRv2Engine(config, backend_factory=factory, model_identifier="model-1")


def test_result_is_immutable_and_propagates_metadata(config):
    result = make_engine(config, Backend()).recognize(candidate())
    assert (result.camera_id, result.track_id, result.frame_index) == ("A", 1, 1)
    assert result.raw_text == " ka-02 mm 9091 "
    assert result.normalized_text == "KA02MM9091"
    assert result.engine_version == "model-1"
    with pytest.raises(FrozenInstanceError): result.raw_text = "changed"


def test_result_converts_to_lightweight_evidence(config):
    result = make_engine(config, Backend()).recognize(candidate())
    evidence = result_to_evidence(result)
    assert evidence.plate_quality_score == 0.85
    assert evidence.raw_text == result.raw_text
    assert evidence.camera_id == "A" and evidence.track_id == 1
    assert not hasattr(evidence, "crop")


def test_model_loads_once_and_recognition_does_not_reload(config):
    backend, loads = Backend(), []
    engine = make_engine(config, backend, loads)
    engine.recognize(candidate(1)); engine.recognize(candidate(2))
    assert loads == [1]
    assert backend.calls == 2


def test_empty_and_inference_failure_are_explicit(config):
    empty = make_engine(config, Backend([{"text": " - ", "score": .7}])).recognize(candidate())
    failed = make_engine(config, Backend(error=RuntimeError("boom"))).recognize(candidate())
    assert empty.status is OCRResultStatus.REJECTED
    assert empty.rejected_reason == "empty_ocr_result"
    assert result_to_evidence(empty) is None
    assert failed.status is OCRResultStatus.FAILED
    assert failed.rejected_reason == "model_inference_error:RuntimeError"


def test_invalid_crop_is_rejected_without_backend_call(config):
    backend = Backend()
    result = make_engine(config, backend).recognize(candidate(crop=np.array([])))
    assert result.rejected_reason == "invalid_crop" and backend.calls == 0


def test_coordinator_preserves_tracks_and_fuses_multiple_frames(config):
    coordinator = TrackOCRCoordinator(make_engine(config, Backend()), MultiFrameOCRFuser())
    one = coordinator.add_candidate(candidate(1, "A", 7))
    assert coordinator.fuser.get("A", 8).status is FusionStatus.UNKNOWN
    assert coordinator.fuser.get("B", 7).status is FusionStatus.UNKNOWN
    assert coordinator.fuser.get("A", 7).status is FusionStatus.UNKNOWN
    coordinator.add_candidate(candidate(2, "A", 7))
    result = coordinator.finalize_track("A", 7)
    assert one.raw_text == " ka-02 mm 9091 "
    assert result.status is FusionStatus.ACCEPTED
    assert result.plate_text == "KA02MM9091"


def test_config_validates_device_and_paths(tmp_path):
    with pytest.raises(ValueError):
        SVTRv2Config(tmp_path, tmp_path, tmp_path, device="gpu")
    with pytest.raises(FileNotFoundError):
        SVTRv2Config(tmp_path / "a", tmp_path / "b", tmp_path / "c").validate_paths()
