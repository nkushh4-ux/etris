from collections.abc import Sequence
from pathlib import Path
from typing import ClassVar

import torch

from cv.common.frame import FramePacket
from cv.detection.detector import VehicleDetector
from cv.detection.models import BoundingBox, VehicleDetection
from ultralytics import YOLO


class UltralyticsVehicleDetector(VehicleDetector):
    """Vehicle detector backed by an Ultralytics YOLO model."""

    VEHICLE_CLASSES : ClassVar[set[str]]={
        "car",
        "motorcycle",
        "bus",
        "truck",
    }

    def __init__(
        self,
        weights: str | Path,
        confidence_threshold: float = 0.25,
        device: str = "cuda",
        half_precision: bool = False,
    ) -> None:
        self.weights = Path(weights)
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.half_precision = half_precision and str(device).startswith("cuda") and torch.cuda.is_available()

        if not self.weights.is_file():
            raise FileNotFoundError(
                f"Model weights not found: {self.weights}"
            )

        self.model = YOLO(str(self.weights))

    def detect(self, frame: FramePacket) -> Sequence[VehicleDetection]:
        """Detect vehicles in one frame."""

        arguments = dict(source=frame.image, conf=self.confidence_threshold,
                         device=self.device, verbose=False)
        if getattr(self, "half_precision", False):
            arguments["half"] = True
        with torch.inference_mode():
            results = self.model.predict(**arguments)

        detections: list[VehicleDetection] = []

        for result in results:
            if result.boxes is None:
                continue

            names = result.names

            for box in result.boxes:
                class_id = int(box.cls.item())
                class_name = str(names[class_id])

                if class_name not in self.VEHICLE_CLASSES:
                    continue

                confidence = float(box.conf.item())
                coordinates = box.xyxy[0].tolist()

                detections.append(
                    VehicleDetection(
                        frame=frame,
                        bbox=BoundingBox(
                            x1=float(coordinates[0]),
                            y1=float(coordinates[1]),
                            x2=float(coordinates[2]),
                            y2=float(coordinates[3]),
                        ),
                        class_name=class_name,
                        confidence=confidence,
                        detector_name="ultralytics",
                        detector_version=str(
                            getattr(self.model, "task", "unknown")
                        ),
                    )
                )

        return detections
