"""Run the Stage 8 smoke path with non-behavioral funnel diagnostics."""

import argparse
import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.config.loader import load_model_config
from cv.ocr.models import OCRResultStatus
from cv.pipeline.models import CompletedTrackANPR

if __package__:
    from scripts.run_anpr_smoke import (
        DEFAULT_CAMERA_ID,
        DEFAULT_VIDEO,
        build_pipeline,
        resolve_path,
    )
else:
    from run_anpr_smoke import (
        DEFAULT_CAMERA_ID,
        DEFAULT_VIDEO,
        build_pipeline,
        resolve_path,
    )
from cv.ingestion.video import VideoFileSource

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "runs/anpr_smoke/stage8_diagnostics.json"
DEFAULT_CROP_DIR = ROOT / "runs/anpr_smoke/stage8_diagnostics/crops"
DEFAULT_CONTACT_SHEET = (
    ROOT / "runs/anpr_smoke/stage8_diagnostics/contact_sheet.jpg"
)


def new_funnel() -> dict[str, Any]:
    return {
        "frames_observed": 0,
        "plate_detections_total": 0,
        "plate_quality_valid": 0,
        "plate_quality_rejected": 0,
        "quality_rejection_reasons": Counter(),
        "stage3_candidates_considered": 0,
        "stage3_candidates_retained": 0,
        "stage3_candidates_replaced": 0,
        "final_selected_candidates": 0,
        "primary_selected_candidates": 0,
        "fallback_selected_candidates": 0,
        "ocr_attempts": 0,
        "ocr_successes": 0,
        "ocr_failures": 0,
        "fusion_status": None,
    }


class DiagnosticCollector:
    def __init__(self, crop_dir: Path | None = None) -> None:
        self.funnels: defaultdict[tuple[str, int], dict[str, Any]] = defaultdict(
            new_funnel
        )
        self.histories: defaultdict[tuple[str, int], list[dict[str, Any]]] = (
            defaultdict(list)
        )
        self.crop_dir = crop_dir
        self.selected_crops: list[dict[str, Any]] = []

    def observe_tracks(self, tracks) -> None:
        for track in tracks:
            key = (track.camera_id, track.track_id)
            self.funnels[key]["frames_observed"] += 1
            self.histories[key].append({
                "frame_index": track.last_frame_index,
                "bbox": [
                    track.bbox.x1, track.bbox.y1,
                    track.bbox.x2, track.bbox.y2,
                ],
                "detection_confidence": track.detection_confidence,
            })

    def record_completed(self, item: CompletedTrackANPR) -> None:
        funnel = self.funnels[(item.camera_id, item.track_id)]
        funnel["final_selected_candidates"] = item.selected_candidate_count
        funnel["ocr_attempts"] = item.selected_candidate_count
        successes = sum(
            result.status is OCRResultStatus.SUCCESS for result in item.ocr_results
        )
        funnel["ocr_successes"] = successes
        funnel["ocr_failures"] = item.selected_candidate_count - successes
        funnel["fusion_status"] = item.fused_plate_result.status.name

    def save_candidate(self, candidate) -> None:
        if self.crop_dir is None:
            return
        self.crop_dir.mkdir(parents=True, exist_ok=True)
        camera = re.sub(r"[^A-Za-z0-9_.-]", "_", candidate.camera_id)
        filename = (
            f"{camera}_track{candidate.track_id}_frame{candidate.frame_index}_"
            f"q{candidate.quality_score:.4f}.jpg"
        )
        path = self.crop_dir / filename
        if not cv2.imwrite(str(path), candidate.crop):
            raise RuntimeError(f"Failed to write selected crop: {path}")
        self.selected_crops.append({
            "camera_id": candidate.camera_id,
            "track_id": candidate.track_id,
            "frame_index": candidate.frame_index,
            "quality_score": candidate.quality_score,
            "selection_tier": candidate.selection_tier,
            "path": str(path.resolve()),
        })


def quality_rejection_reasons(detection, config) -> tuple[str, ...]:
    """Mirror Stage 3 hard guards for diagnostics without changing scoring."""

    image = detection.frame.image
    if image.size == 0 or image.ndim not in (2, 3):
        return ("invalid_or_empty_frame",)
    height, width = image.shape[:2]
    box = detection.bbox
    requested_area = box.width * box.height
    if requested_area <= 0.0:
        return ("invalid_bbox",)
    x1 = max(0, min(width, int(np.floor(box.x1))))
    y1 = max(0, min(height, int(np.floor(box.y1))))
    x2 = max(0, min(width, int(np.ceil(box.x2))))
    y2 = max(0, min(height, int(np.ceil(box.y2))))
    crop_width, crop_height = x2 - x1, y2 - y1
    reasons = []
    fallback_minimum = getattr(
        config, "fallback_min_plate_width_px", config.min_plate_width_px
    )
    if crop_width < fallback_minimum:
        reasons.append("width_below_minimum")
    if crop_height < config.min_plate_height_px:
        reasons.append("height_below_minimum")
    if reasons:
        return tuple(reasons)
    visible_fraction = (crop_width * crop_height) / requested_area
    if visible_fraction < config.min_visible_fraction:
        return ("clipping_below_minimum_visible_fraction",)
    if image[y1:y2, x1:x2].size == 0:
        return ("invalid_or_empty_crop",)
    return ("other_configured_hard_rejection",)


class DiagnosticPlateDetector:
    def __init__(self, delegate, collector: DiagnosticCollector) -> None:
        self.delegate = delegate
        self.collector = collector

    def detect_in_frame(self, track, image, frame_packet):
        detections = self.delegate.detect_in_frame(track, image, frame_packet)
        self.collector.funnels[(track.camera_id, track.track_id)][
            "plate_detections_total"
        ] += len(detections)
        return detections


class DiagnosticQualityAssessor:
    def __init__(self, delegate, collector: DiagnosticCollector) -> None:
        self.delegate = delegate
        self.config = delegate.config
        self.collector = collector

    def assess(self, detection):
        candidate = self.delegate.assess(detection)
        key = (detection.frame.camera_id, detection.track_id)
        funnel = self.collector.funnels[key]
        if candidate is not None:
            funnel["plate_quality_valid"] += 1
        else:
            funnel["plate_quality_rejected"] += 1
            funnel["quality_rejection_reasons"].update(
                quality_rejection_reasons(detection, self.config)
            )
        return candidate


class DiagnosticSelector:
    def __init__(self, delegate, collector: DiagnosticCollector) -> None:
        self.delegate = delegate
        self.collector = collector

    def offer(self, candidate):
        key = (candidate.camera_id, candidate.track_id)
        funnel = self.collector.funnels[key]
        funnel["stage3_candidates_considered"] += 1
        before = {id(item) for item in self.delegate.get(*key)}
        retained = self.delegate.offer(candidate)
        after = {id(item) for item in self.delegate.get(*key)}
        if retained:
            funnel["stage3_candidates_retained"] += 1
        funnel["stage3_candidates_replaced"] += len(before - after)
        return retained

    def get(self, camera_id, track_id):
        return self.delegate.get(camera_id, track_id)

    def finalize(self, camera_id, track_id):
        candidates = self.delegate.finalize(camera_id, track_id)
        funnel = self.collector.funnels[(camera_id, track_id)]
        funnel["primary_selected_candidates"] += sum(
            item.selection_tier == "PRIMARY" for item in candidates
        )
        funnel["fallback_selected_candidates"] += sum(
            item.selection_tier == "FALLBACK" for item in candidates
        )
        for candidate in candidates:
            self.collector.save_candidate(candidate)
        return candidates


def bbox_metrics(first: list[float], second: list[float]) -> tuple[float, float]:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    iou = intersection / union if union else 0.0
    center_distance = math.hypot(
        (ax1 + ax2 - bx1 - bx2) / 2.0,
        (ay1 + ay2 - by1 - by2) / 2.0,
    )
    return iou, center_distance


def compare_track_pairs(histories) -> list[dict[str, Any]]:
    pairs = []
    keys = sorted(key for key, history in histories.items() if history)
    for index, first_key in enumerate(keys):
        for second_key in keys[index + 1:]:
            if first_key[0] != second_key[0]:
                continue
            first = {item["frame_index"]: item for item in histories[first_key]}
            second = {item["frame_index"]: item for item in histories[second_key]}
            first_span, second_span = (min(first), max(first)), (min(second), max(second))
            overlap = max(
                0, min(first_span[1], second_span[1])
                - max(first_span[0], second_span[0]) + 1,
            )
            gap = max(
                0, max(first_span[0], second_span[0])
                - min(first_span[1], second_span[1]) - 1,
            )
            if overlap == 0 and gap > 1:
                continue
            shared = sorted(set(first) & set(second))
            shared_metrics = []
            for frame_index in shared:
                iou, distance = bbox_metrics(
                    first[frame_index]["bbox"], second[frame_index]["bbox"]
                )
                shared_metrics.append({
                    "frame_index": frame_index,
                    "bbox_iou": iou,
                    "center_distance_px": distance,
                })
            nearest_frames = min(
                ((abs(a - b), a, b) for a in first for b in second),
                key=lambda value: (value[0], value[1], value[2]),
            )
            nearest_iou, nearest_distance = bbox_metrics(
                first[nearest_frames[1]]["bbox"],
                second[nearest_frames[2]]["bbox"],
            )
            pairs.append({
                "camera_id": first_key[0],
                "track_a": first_key[1],
                "track_b": second_key[1],
                "track_a_span": list(first_span),
                "track_b_span": list(second_span),
                "temporal_overlap_frames": overlap,
                "adjacent_gap_frames": gap,
                "coexist_in_same_observed_frame": bool(shared),
                "shared_frame_metrics": shared_metrics,
                "nearest_observed_frames": {
                    "track_a_frame": nearest_frames[1],
                    "track_b_frame": nearest_frames[2],
                    "frame_distance": nearest_frames[0],
                    "bbox_iou": nearest_iou,
                    "center_distance_px": nearest_distance,
                },
            })
    return pairs


def fusion_diagnostics(item: CompletedTrackANPR) -> dict[str, Any] | None:
    fused = item.fused_plate_result
    if fused.status.name not in {"ACCEPTED", "LOW_CONFIDENCE"}:
        return None
    constraint = fused.constraint_diagnostic
    return {
        "camera_id": item.camera_id,
        "track_id": item.track_id,
        "candidate_plate": fused.plate_text,
        "final_confidence": fused.confidence,
        "cluster_weight_share": fused.dominant_cluster_weight_share,
        "mean_position_confidence": fused.mean_position_confidence,
        "minimum_position_confidence": fused.minimum_position_confidence,
        "india_constraint": None if constraint is None else {
            "raw_consensus_text": constraint.raw_consensus_text,
            "constrained_text": constraint.constrained_text,
            "format": constraint.format.value,
            "prefix": constraint.prefix,
            "prefix_status": constraint.prefix_status.value,
            "constraint_applied": constraint.constraint_applied,
            "changed_positions": list(constraint.changed_positions),
            "candidate_score": constraint.candidate_score,
            "raw_score": constraint.raw_score,
            "score_margin": constraint.score_margin,
            "reason_codes": list(constraint.reason_codes),
        },
        "characters": [
            {
                "position": character.position,
                "selected_character": character.character,
                "weighted_support": character.weighted_support,
                "total_compatible_weight": character.total_compatible_weight,
                "position_confidence": character.position_confidence,
                "supporting_unique_frames": character.supporting_frame_count,
                "weighted_alternatives": None,
            }
            for character in fused.character_diagnostics
        ],
        "alternatives_note": (
            "Weighted alternatives are not exposed by the locked Stage 5 result model."
        ),
    }


def create_contact_sheet(crops: list[dict[str, Any]], output: Path) -> None:
    if not crops:
        return
    cell_width, cell_height, columns = 260, 130, 5
    rows = math.ceil(len(crops) / columns)
    sheet = np.full((rows * cell_height, columns * cell_width, 3), 245,
                    dtype=np.uint8)
    for index, metadata in enumerate(crops):
        image = cv2.imread(metadata["path"], cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to read selected crop: {metadata['path']}")
        scale = min(240 / image.shape[1], 90 / image.shape[0])
        resized = cv2.resize(image, None, fx=scale, fy=scale)
        row, column = divmod(index, columns)
        x = column * cell_width + 10
        y = row * cell_height + 25
        sheet[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        label = (
            f"{metadata['camera_id']} T{metadata['track_id']} "
            f"F{metadata['frame_index']} Q{metadata['quality_score']:.3f}"
            f" {metadata['selection_tier']}_SELECTED"
        )
        cv2.putText(sheet, label, (x, row * cell_height + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, .38, (0, 0, 0), 1, cv2.LINE_AA)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), sheet):
        raise RuntimeError(f"Failed to write contact sheet: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--camera-id", default=DEFAULT_CAMERA_ID)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--save-selected-crops", action="store_true")
    args = parser.parse_args()
    if args.max_frames < 1:
        parser.error("--max-frames must be at least 1")

    video, output = resolve_path(args.video), resolve_path(args.output)
    crop_dir = DEFAULT_CROP_DIR if args.save_selected_crops else None
    collector = DiagnosticCollector(crop_dir)
    pipeline, _ = build_pipeline(load_model_config())
    original_selector = pipeline.best_frame_selector
    pipeline.plate_detector = DiagnosticPlateDetector(
        pipeline.plate_detector, collector
    )
    pipeline.quality_assessor = DiagnosticQualityAssessor(
        pipeline.quality_assessor, collector
    )
    pipeline.best_frame_selector = DiagnosticSelector(
        original_selector, collector
    )

    completed: list[CompletedTrackANPR] = []
    frames_processed = 0
    started = time.perf_counter()
    for packet in VideoFileSource(video, args.camera_id):
        tracks = pipeline.process_frame(packet)
        collector.observe_tracks(tracks)
        emitted = pipeline.pop_completed_tracks()
        for item in emitted:
            collector.record_completed(item)
        completed.extend(emitted)
        frames_processed += 1
        if frames_processed >= args.max_frames:
            break
    emitted = pipeline.finalize()
    for item in emitted:
        collector.record_completed(item)
    completed.extend(emitted)
    elapsed = time.perf_counter() - started

    if args.save_selected_crops:
        create_contact_sheet(collector.selected_crops, DEFAULT_CONTACT_SHEET)

    totals = Counter()
    per_track = []
    completed_by_key = {(item.camera_id, item.track_id): item for item in completed}
    for key in sorted(collector.funnels):
        funnel = dict(collector.funnels[key])
        funnel["quality_rejection_reasons"] = dict(
            funnel["quality_rejection_reasons"]
        )
        for name in (
            "plate_detections_total", "plate_quality_valid",
            "plate_quality_rejected", "final_selected_candidates",
            "primary_selected_candidates", "fallback_selected_candidates",
            "ocr_attempts", "ocr_successes", "ocr_failures",
        ):
            totals[name] += funnel[name]
        history = collector.histories.get(key, [])
        terminal = completed_by_key.get(key)
        per_track.append({
            "camera_id": key[0],
            "track_id": key[1],
            **funnel,
            "first_frame": history[0]["frame_index"] if history else None,
            "last_frame": history[-1]["frame_index"] if history else None,
            "observed_history": history,
            "termination_reason": (
                terminal.termination_reason.name if terminal else None
            ),
        })

    report = {
        "global": {
            "video_path": str(video),
            "frame_limit": args.max_frames,
            "frames_processed": frames_processed,
            "completed_tracks": len(completed),
            "elapsed_seconds": elapsed,
            "processing_fps": frames_processed / elapsed if elapsed else 0.0,
        },
        "funnel_totals": dict(totals),
        "per_track_funnel": per_track,
        "track_pair_overlap_diagnostics": compare_track_pairs(collector.histories),
        "fusion_character_diagnostics": [
            diagnostic for item in completed
            if (diagnostic := fusion_diagnostics(item)) is not None
        ],
        "selected_crop_artifacts": collector.selected_crops,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(f"Diagnostic report: {output}")
    if args.save_selected_crops:
        print(f"Selected crops: {DEFAULT_CROP_DIR}")
        print(f"Contact sheet: {DEFAULT_CONTACT_SHEET}")


if __name__ == "__main__":
    main()
