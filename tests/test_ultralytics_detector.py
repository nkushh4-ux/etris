from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np

from cv.common.frame import FramePacket
from cv.detection.ultralytics_detector import UltralyticsVehicleDetector


class FakeBox:
    def __init__(
        self,
        class_id: int,
        confidence: float,
        coordinates: list[float],
    ) -> None:
        self.cls = SimpleNamespace(item=lambda: class_id)
        self.conf = SimpleNamespace(item=lambda: confidence)
        self.xyxy = [SimpleNamespace(tolist=lambda: coordinates)]


class FakeModel:
    task = "detect"

    def predict(self, **kwargs):
        return [
            SimpleNamespace(
                names={
                    0: "car",
                    1: "person",
                },
                boxes=[
                    FakeBox(0, 0.91, [10, 20, 110, 120]),
                    FakeBox(1, 0.99, [50, 50, 80, 100]),
                ],
            )
        ]


def test_detector_returns_only_vehicle_classes(monkeypatch, tmp_path):
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"fake")

    detector = UltralyticsVehicleDetector.__new__(
        UltralyticsVehicleDetector
    )

    detector.weights = weights
    detector.confidence_threshold = 0.25
    detector.device = "cpu"
    detector.model = FakeModel()

    frame = FramePacket(
        camera_id="CAM-001",
        frame_index=0,
        timestamp=datetime.now(UTC),
        image=np.zeros((240, 320, 3), dtype=np.uint8),
    )

    detections = detector.detect(frame)

    assert len(detections) == 1

    detection = detections[0]

    assert detection.class_name == "car"
    assert detection.confidence == 0.91
    assert detection.bbox.x1 == 10
    assert detection.bbox.y1 == 20
    assert detection.bbox.x2 == 110
    assert detection.bbox.y2 == 120
    assert detection.frame.camera_id == "CAM-001"


def test_detector_rejects_missing_weights(tmp_path):
    missing_weights = tmp_path / "missing.pt"

    try:
        UltralyticsVehicleDetector(
            weights=missing_weights,
            device="cpu",
        )
    except FileNotFoundError as exc:
        assert "Model weights not found" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError")