"""Collect bounded plate-width samples without changing production decisions."""

from __future__ import annotations

import argparse
import csv
import math
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.config.loader import load_model_config
from cv.ingestion.video import VideoFileSource
from cv.ocr.svtrv2 import SVTRv2Config, SVTRv2Engine
from cv.plate.models import PlateCandidate, PlateDetection
from cv.plate.quality import PlateQualityAssessor, PlateQualityConfig
from cv.tracking.factory import create_tracker
from cv.tracking.types import TrackingAlgorithm

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = ROOT / "cv/data/video/traffic.mp4"
DEFAULT_CAMERA_ID = "cam_smoke_01"
DEFAULT_OUTPUT_DIR = ROOT / "runs/anpr_smoke/width_calibration"
WIDTH_BANDS = ("<40", "40-49", "50-59", "60-69", "70-79", "80-99", "100+")


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def quality_rejection_reasons(detection: PlateDetection, config: PlateQualityConfig) -> tuple[str, ...]:
    """Mirror the production assessor's hard guards without changing them."""
    image = detection.frame.image
    if image.size == 0 or image.ndim not in (2, 3):
        return ("invalid_or_empty_frame",)
    image_height, image_width = image.shape[:2]
    box = detection.bbox
    requested_area = box.width * box.height
    if requested_area <= 0:
        return ("invalid_bbox",)
    x1, y1 = max(0, min(image_width, math.floor(box.x1))), max(0, min(image_height, math.floor(box.y1)))
    x2, y2 = max(0, min(image_width, math.ceil(box.x2))), max(0, min(image_height, math.ceil(box.y2)))
    width, height = x2 - x1, y2 - y1
    reasons = []
    if width < config.min_plate_width_px:
        reasons.append("width_below_minimum")
    if height < config.min_plate_height_px:
        reasons.append("height_below_minimum")
    if reasons:
        return tuple(reasons)
    if (width * height) / requested_area < config.min_visible_fraction:
        return ("clipping_below_minimum_visible_fraction",)
    if image[y1:y2, x1:x2].size == 0:
        return ("invalid_or_empty_crop",)
    return ("other_configured_hard_rejection",)


def width_band(width_px: int) -> str:
    if width_px < 40:
        return "<40"
    if width_px < 50:
        return "40-49"
    if width_px < 60:
        return "50-59"
    if width_px < 70:
        return "60-69"
    if width_px < 80:
        return "70-79"
    if width_px < 100:
        return "80-99"
    return "100+"


@dataclass(slots=True)
class Observation:
    detection: PlateDetection
    candidate: PlateCandidate
    production_valid: bool
    rejection_reason: str

    @property
    def width_px(self) -> int:
        return self.candidate.metrics.crop_width_px

    @property
    def band(self) -> str:
        return width_band(self.width_px)


def select_representatives(
    observations: Iterable[Observation], *, maximum: int = 15, min_frame_gap: int = 5
) -> list[Observation]:
    """Deterministically favor track diversity, spacing, then crop quality."""
    grouped: dict[str, list[Observation]] = defaultdict(list)
    for item in observations:
        grouped[item.band].append(item)

    selected: list[Observation] = []
    for band in WIDTH_BANDS:
        by_track: dict[int, list[Observation]] = defaultdict(list)
        items = sorted(
            grouped[band],
            key=lambda item: (
                item.detection.track_id,
                -item.candidate.quality_score,
                item.detection.frame.frame_index,
            ),
        )
        for item in items:
            accepted = by_track[item.detection.track_id]
            if not accepted or all(
                abs(item.detection.frame.frame_index - prior.detection.frame.frame_index)
                >= min_frame_gap
                for prior in accepted
            ):
                accepted.append(item)

        queues = {
            track_id: sorted(values, key=lambda item: (-item.candidate.quality_score, item.detection.frame.frame_index))
            for track_id, values in by_track.items()
        }
        band_selected: list[Observation] = []
        while len(band_selected) < maximum and any(queues.values()):
            for track_id in sorted(queues):
                if queues[track_id] and len(band_selected) < maximum:
                    band_selected.append(queues[track_id].pop(0))
        selected.extend(band_selected)
    return selected


def _required(config: dict[str, Any], name: str) -> dict[str, Any]:
    section = config.get(name)
    if not isinstance(section, dict) or not section.get("enabled", False):
        raise RuntimeError(f"Required component is disabled: {name}")
    return section


def _components(config: dict[str, Any]):
    # Keep heavyweight Ultralytics initialization out of import-only tests.
    from cv.detection.ultralytics_detector import UltralyticsVehicleDetector
    from cv.plate.ultralytics_plate_detector import UltralyticsPlateDetector

    vehicle = _required(config, "vehicle_detection")
    plate = _required(config, "plate_detection")
    tracking = _required(config, "tracking")
    quality_values = dict(_required(config, "quality_assessment"))
    quality_values.pop("enabled", None)
    production_config = PlateQualityConfig.from_mapping(quality_values)

    detector = UltralyticsVehicleDetector(
        resolve_path(vehicle["weights"]),
        float(vehicle["confidence_threshold"]),
        str(vehicle["device"]),
    )
    plate_args: dict[str, Any] = {
        "weights": resolve_path(plate["weights"]),
        "confidence_threshold": float(plate["confidence_threshold"]),
        "device": str(plate["device"]),
    }
    if plate.get("input_size") is not None:
        plate_args["input_size"] = int(plate["input_size"])
    return (
        detector,
        create_tracker(TrackingAlgorithm(tracking["algorithm"])),
        UltralyticsPlateDetector(**plate_args),
        production_config,
        PlateQualityAssessor(production_config),
        PlateQualityAssessor(PlateQualityConfig.from_mapping({
            **quality_values,
            "min_plate_width_px": 1,
            "fallback_min_plate_width_px": 1,
        })),
    )


def _detection_row(detection: PlateDetection, candidate: PlateCandidate | None,
                   valid: bool, reason: str) -> dict[str, Any]:
    image_height, image_width = detection.frame.image.shape[:2]
    box = detection.bbox
    x1 = max(0, min(image_width, math.floor(box.x1)))
    y1 = max(0, min(image_height, math.floor(box.y1)))
    x2 = max(0, min(image_width, math.ceil(box.x2)))
    y2 = max(0, min(image_height, math.ceil(box.y2)))
    width, height = x2 - x1, y2 - y1
    row = {
        "camera_id": detection.frame.camera_id,
        "track_id": detection.track_id,
        "frame_index": detection.frame.frame_index,
        "plate_bbox": f"{box.x1:.3f},{box.y1:.3f},{box.x2:.3f},{box.y2:.3f}",
        "width_px": width,
        "height_px": height,
        "detector_confidence": detection.confidence,
        "quality_score": "" if candidate is None else candidate.quality_score,
        "width_band": width_band(width),
        "production_valid": valid,
        "production_rejection_reason": reason,
    }
    for name in ("sharpness", "exposure", "contrast", "size", "detector_confidence", "clipping"):
        row[f"quality_{name}"] = "" if candidate is None else getattr(candidate.metrics, name)
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _contact_sheet(samples: list[dict[str, Any]], path: Path) -> None:
    if not samples:
        return
    thumb_w, thumb_h, label_h, columns = 260, 110, 34, 3
    rows = (len(samples) + columns - 1) // columns
    canvas = 255 * np.ones((rows * (thumb_h + label_h), columns * thumb_w, 3), dtype="uint8")
    for index, sample in enumerate(samples):
        image = cv2.imread(sample["crop_path"])
        if image is None:
            continue
        scale = min(thumb_w / image.shape[1], thumb_h / image.shape[0])
        resized = cv2.resize(image, (max(1, int(image.shape[1] * scale)), max(1, int(image.shape[0] * scale))))
        row, column = divmod(index, columns)
        x = column * thumb_w + (thumb_w - resized.shape[1]) // 2
        y = row * (thumb_h + label_h) + (thumb_h - resized.shape[0]) // 2
        canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        label = f"{sample['sample_id']} T{sample['track_id']} F{sample['frame_index']} W{sample['width_px']} Q{float(sample['quality_score']):.2f}"
        cv2.putText(canvas, label, (column * thumb_w + 4, row * (thumb_h + label_h) + thumb_h + 22), cv2.FONT_HERSHEY_SIMPLEX, .43, (0, 0, 0), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(path), canvas):
        raise RuntimeError(f"Failed to write contact sheet: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--camera-id", default=DEFAULT_CAMERA_ID)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--max-per-band", type=int, default=15)
    parser.add_argument("--min-frame-gap", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-ocr", action="store_true")
    args = parser.parse_args()
    if args.max_frames < 1 or args.max_per_band < 1 or args.min_frame_gap < 0:
        parser.error("frame/sample limits must be positive and min-frame-gap non-negative")

    output_dir = resolve_path(args.output_dir)
    crop_dir = output_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)
    detector, tracker, plate_detector, production_config, production_assessor, diagnostic_assessor = _components(load_model_config())

    observations: list[Observation] = []
    detection_rows: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    for frame in VideoFileSource(resolve_path(args.video), args.camera_id):
        if frame.frame_index >= args.max_frames:
            break
        tracks = tracker.update(detector.detect(frame))
        tracker.pop_expired_tracks()
        for track in tracks:
            for detection in plate_detector.detect_in_frame(track, frame.image, frame):
                production_candidate = production_assessor.assess(detection)
                reasons = () if production_candidate is not None else quality_rejection_reasons(detection, production_config)
                reason = "" if not reasons else ";".join(reasons)
                candidate = production_candidate or diagnostic_assessor.assess(detection)
                row = _detection_row(detection, candidate, production_candidate is not None, reason)
                detection_rows.append(row)
                totals[row["width_band"]] += 1
                if candidate is not None:
                    observations.append(Observation(detection, candidate, production_candidate is not None, reason))
    tracker.finalize_all()

    selected = select_representatives(observations, maximum=args.max_per_band, min_frame_gap=args.min_frame_gap)
    engine = None
    if args.run_ocr:
        ocr = _required(load_model_config(), "ocr")
        svtr_values = dict(_required(ocr, "svtrv2_s"))
        svtr_values.pop("enabled", None)
        engine = SVTRv2Engine(SVTRv2Config.from_mapping(svtr_values))

    manifest_rows: list[dict[str, Any]] = []
    sampled: Counter[str] = Counter()
    nonempty: Counter[str] = Counter()
    confidence_sum: Counter[str] = Counter()
    for number, item in enumerate(selected, 1):
        sample_id = f"S{number:03d}"
        filename = f"{sample_id}_track{item.detection.track_id}_frame{item.detection.frame.frame_index}_w{item.width_px}_q{item.candidate.quality_score:.3f}.jpg"
        crop_path = crop_dir / filename
        if not cv2.imwrite(str(crop_path), item.candidate.crop):
            raise RuntimeError(f"Failed to write crop: {crop_path}")
        raw_text = normalized_text = ocr_confidence = ocr_latency_ms = ""
        if engine is not None:
            started = time.perf_counter()
            result = engine.recognize(item.candidate)
            ocr_latency_ms = (time.perf_counter() - started) * 1000.0
            raw_text, normalized_text, ocr_confidence = result.raw_text, result.normalized_text, result.confidence
            confidence_sum[item.band] += result.confidence
            if result.normalized_text:
                nonempty[item.band] += 1
        sampled[item.band] += 1
        manifest_rows.append({
            "sample_id": sample_id, "camera_id": item.detection.frame.camera_id,
            "track_id": item.detection.track_id, "frame_index": item.detection.frame.frame_index,
            "crop_path": str(crop_path.resolve()), "width_px": item.width_px,
            "height_px": item.candidate.metrics.crop_height_px,
            "detector_confidence": item.detection.confidence,
            "quality_score": item.candidate.quality_score, "width_band": item.band,
            "production_valid": item.production_valid,
            "production_rejection_reason": item.rejection_reason,
            "raw_text": raw_text, "normalized_text": normalized_text,
            "ocr_confidence": ocr_confidence, "ocr_latency_ms": ocr_latency_ms,
            "ground_truth": "", "label_status": "",
        })

    detection_fields = list(detection_rows[0]) if detection_rows else [
        "camera_id", "track_id", "frame_index", "plate_bbox", "width_px", "height_px",
        "detector_confidence", "quality_score", "width_band", "production_valid",
        "production_rejection_reason", "quality_sharpness", "quality_exposure",
        "quality_contrast", "quality_size", "quality_detector_confidence", "quality_clipping",
    ]
    manifest_fields = list(manifest_rows[0]) if manifest_rows else [
        "sample_id", "camera_id", "track_id", "frame_index", "crop_path", "width_px",
        "height_px", "detector_confidence", "quality_score", "width_band",
        "production_valid", "production_rejection_reason", "raw_text", "normalized_text",
        "ocr_confidence", "ocr_latency_ms", "ground_truth", "label_status",
    ]
    _write_csv(output_dir / "detections.csv", detection_rows, detection_fields)
    _write_csv(output_dir / "manifest.csv", manifest_rows, manifest_fields)
    _contact_sheet(manifest_rows, output_dir / "contact_sheet.jpg")

    print("width_band,total_detections,sampled_crops" + (",non_empty_ocr,mean_ocr_confidence" if engine else ""))
    for band in WIDTH_BANDS:
        line = f"{band},{totals[band]},{sampled[band]}"
        if engine:
            mean = confidence_sum[band] / sampled[band] if sampled[band] else 0.0
            line += f",{nonempty[band]},{mean:.4f}"
        print(line)
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
