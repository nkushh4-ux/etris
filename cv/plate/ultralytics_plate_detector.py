from collections.abc import Sequence
from pathlib import Path

import torch

from cv.detection.models import BoundingBox
from cv.plate.detector import PlateDetector
from cv.plate.models import PlateDetection
from cv.tracking.models import TrackState
from ultralytics import YOLO


class UltralyticsPlateDetector(PlateDetector):
    """Indian license-plate detector backed by an Ultralytics model."""

    def __init__(
        self,
        weights: str | Path,
        confidence_threshold: float = 0.25,
        device: str = "cuda",
        input_size: int = 640,
        roi_padding: float = 0.10,
        half_precision: bool = False,
        max_batch_size: int = 4,
    ) -> None:
        self.weights = Path(weights)
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.input_size = input_size
        self.roi_padding = roi_padding
        self.half_precision = half_precision and str(device).startswith("cuda") and torch.cuda.is_available()
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        self.max_batch_size = max_batch_size
        self.model_invocations = 0
        self.rois_processed = 0

        if not self.weights.is_file():
            raise FileNotFoundError(
                f"Plate detector weights not found: {self.weights}"
            )

        if not 0.0 <= roi_padding <= 1.0:
            raise ValueError(
                "roi_padding must be between 0 and 1"
            )

        self.model = YOLO(str(self.weights))

    def detect(
        self,
        track: TrackState,
    ) -> Sequence[PlateDetection]:
        """Detect plates inside the current vehicle ROI."""

        frame = track.bbox
        image = track.history[-1] if False else None

        # TrackState intentionally does not store the image.
        # The frame image is recovered from the latest detection
        # only after the pipeline supplies it.
        raise NotImplementedError(
            "Plate detection requires the current FramePacket. "
            "Use detect_in_frame() from the pipeline adapter."
        )

    def detect_in_frame(
        self,
        track: TrackState,
        image,
        frame_packet,
    ) -> Sequence[PlateDetection]:
        """Detect plates using the current frame image."""

        return self.detect_batch_in_frame((track,), image, frame_packet)

    def detect_batch_in_frame(self, tracks, image, frame_packet) -> Sequence[PlateDetection]:
        """Run bounded batched inference and retain exact ROI-to-track mapping."""
        frame_height, frame_width = image.shape[:2]
        prepared = []
        for track in tracks:
            roi_data = self._prepare_roi(track, image)
            if roi_data is not None:
                roi, x1, y1 = roi_data
                prepared.append((track, roi, x1, y1))

        detections: list[PlateDetection] = []
        for start in range(0, len(prepared), self.max_batch_size):
            batch = prepared[start:start + self.max_batch_size]
            arguments = dict(source=[item[1] for item in batch], conf=self.confidence_threshold,
                             imgsz=self.input_size, device=self.device, verbose=False)
            if self.half_precision:
                arguments["half"] = True
            with torch.inference_mode():
                results = self.model.predict(**arguments)
            self.model_invocations += 1
            self.rois_processed += len(batch)
            for (track, _roi, offset_x, offset_y), result in zip(batch, results, strict=True):
                detections.extend(self._map_result(
                    result, track, frame_packet, offset_x, offset_y, frame_width, frame_height
                ))
        return detections

    def _prepare_roi(self, track, image):
        frame_height, frame_width = image.shape[:2]

        x1 = max(
            0,
            int(track.bbox.x1 - track.bbox.width * self.roi_padding),
        )
        y1 = max(
            0,
            int(track.bbox.y1 - track.bbox.height * self.roi_padding),
        )
        x2 = min(
            frame_width,
            int(track.bbox.x2 + track.bbox.width * self.roi_padding),
        )
        y2 = min(
            frame_height,
            int(track.bbox.y2 + track.bbox.height * self.roi_padding),
        )

        if x2 <= x1 or y2 <= y1:
            return None

        roi = image[y1:y2, x1:x2]

        if roi.size == 0:
            return None
        return roi, x1, y1

    def _map_result(self, result, track, frame_packet, x1, y1, frame_width, frame_height):
        detections: list[PlateDetection] = []
        if result.boxes is not None:
            for box in result.boxes:
                coordinates = box.xyxy[0].tolist()

                local_x1 = float(coordinates[0])
                local_y1 = float(coordinates[1])
                local_x2 = float(coordinates[2])
                local_y2 = float(coordinates[3])

                full_x1 = max(0.0, local_x1 + x1)
                full_y1 = max(0.0, local_y1 + y1)
                full_x2 = min(float(frame_width), local_x2 + x1)
                full_y2 = min(float(frame_height), local_y2 + y1)

                if full_x2 <= full_x1 or full_y2 <= full_y1:
                    continue

                detections.append(
                    PlateDetection(
                        frame=frame_packet,
                        track_id=track.track_id,
                        bbox=BoundingBox(
                            x1=full_x1,
                            y1=full_y1,
                            x2=full_x2,
                            y2=full_y2,
                        ),
                        confidence=float(box.conf.item()),
                        detector_name="ultralytics",
                        detector_version=self.weights.stem,
                    )
                )

        return detections
