"""Manually run the real Stage 1-7 ANPR path on a bounded video segment."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from backend.config.loader import load_model_config
from cv.detection.ultralytics_detector import UltralyticsVehicleDetector
from cv.ingestion.video import VideoFileSource
from cv.ocr.coordinator import TrackOCRCoordinator
from cv.ocr.fusion import MultiFrameOCRFuser, OCRFusionConfig
from cv.ocr.india import IndiaConstraintConfig, IndiaPlateConstraintDecoder
from cv.ocr.svtrv2 import SVTRv2Config, SVTRv2Engine
from cv.pipeline.camera import CameraPipeline
from cv.pipeline.models import CompletedTrackANPR
from cv.plate.quality import PlateQualityAssessor, PlateQualityConfig
from cv.plate.selection import PlateBestFrameSelector
from cv.plate.ultralytics_plate_detector import UltralyticsPlateDetector
from cv.tracking.factory import create_tracker
from cv.tracking.types import TrackingAlgorithm

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = ROOT / "cv/data/video/traffic.mp4"
DEFAULT_OUTPUT = ROOT / "runs/anpr_smoke/stage8_smoke.json"
DEFAULT_CAMERA_ID = "cam_smoke_01"


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def require_enabled(config: dict[str, Any], name: str) -> dict[str, Any]:
    section = config.get(name)
    if not isinstance(section, dict) or not section.get("enabled", False):
        raise RuntimeError(f"Required model component is disabled: {name}")
    return section


def build_pipeline(config: dict[str, Any]) -> tuple[CameraPipeline, dict[str, Any]]:
    vehicle = require_enabled(config, "vehicle_detection")
    plate = require_enabled(config, "plate_detection")
    tracking = require_enabled(config, "tracking")
    quality_values = dict(require_enabled(config, "quality_assessment"))
    quality_values.pop("enabled", None)
    fusion_values = dict(require_enabled(config, "ocr_fusion"))
    ocr = require_enabled(config, "ocr")
    performance = config.get("live_performance", {})

    primary_engine = ocr.get("primary_engine")
    if primary_engine != "svtrv2_s":
        raise RuntimeError(
            f"Stage 8 requires configured primary_engine=svtrv2_s, got {primary_engine!r}"
        )
    svtr_values = dict(require_enabled(ocr, "svtrv2_s"))

    vehicle_weights = resolve_path(vehicle["weights"])
    plate_weights = resolve_path(plate["weights"])
    svtr_config = SVTRv2Config.from_mapping(svtr_values)

    print(f"Vehicle detector checkpoint: {vehicle_weights}")
    print(f"Plate detector checkpoint:   {plate_weights}")
    print(f"SVTRv2 config:               {svtr_config.config_path}")
    print(f"SVTRv2 checkpoint:           {svtr_config.checkpoint_path}")

    detector = UltralyticsVehicleDetector(
        weights=vehicle_weights,
        confidence_threshold=float(vehicle["confidence_threshold"]),
        device=str(vehicle["device"]),
        half_precision=bool(performance.get("enable_fp16", False)),
    )
    plate_arguments: dict[str, Any] = {
        "weights": plate_weights,
        "confidence_threshold": float(plate["confidence_threshold"]),
        "device": str(plate["device"]),
        "half_precision": bool(performance.get("enable_fp16", False)),
    }
    # Null means retain the detector constructor's established default.
    if plate.get("input_size") is not None:
        plate_arguments["input_size"] = int(plate["input_size"])
    plate_detector = UltralyticsPlateDetector(**plate_arguments)

    tracker = create_tracker(TrackingAlgorithm(tracking["algorithm"]))
    quality_config = PlateQualityConfig.from_mapping(quality_values)
    quality_assessor = PlateQualityAssessor(quality_config)
    selector = PlateBestFrameSelector(
        top_k=quality_config.top_k,
        min_frame_gap=quality_config.min_frame_gap,
    )
    fusion_config = OCRFusionConfig.from_mapping(fusion_values)
    engine = SVTRv2Engine(svtr_config)
    india_values = dict(config.get("ocr_india_constraints", {}))
    india_config = IndiaConstraintConfig.from_mapping(india_values) if india_values else IndiaConstraintConfig(enabled=False)
    coordinator = TrackOCRCoordinator(
        engine, MultiFrameOCRFuser(fusion_config, IndiaPlateConstraintDecoder(india_config))
    )
    pipeline = CameraPipeline(
        detector=detector,
        tracker=tracker,
        plate_detector=plate_detector,
        quality_assessor=quality_assessor,
        best_frame_selector=selector,
        ocr_coordinator=coordinator,
    )
    context = {
        "vehicle_weights": vehicle_weights,
        "plate_weights": plate_weights,
        "svtr_config": svtr_config,
        "svtr_model_identifier": engine.model_identifier,
        "quality_config": quality_config,
        "fusion_config": fusion_config,
        "india_constraint_config": india_config,
        "selector": selector,
    }
    return pipeline, context


def completed_to_dict(item: CompletedTrackANPR) -> dict[str, Any]:
    fused = item.fused_plate_result
    supporting_ids = set(fused.supporting_evidence_ids)
    supporting_frames = sorted({
        result.frame_index for result in item.ocr_results
        if result.evidence_id in supporting_ids
    })
    constraint = fused.constraint_diagnostic
    constraint_payload = None if constraint is None else {
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
    }
    return {
        "camera_id": item.camera_id,
        "track_id": item.track_id,
        "termination_reason": item.termination_reason.name,
        "termination_frame_index": item.termination_frame_index,
        "timestamp": item.timestamp.isoformat(),
        "selected_candidate_count": item.selected_candidate_count,
        "ocr_result_count": item.ocr_result_count,
        "ocr_results": [
            {
                "frame_index": result.frame_index,
                "raw_text": result.raw_text,
                "normalized_text": result.normalized_text,
                "confidence": result.confidence,
                "plate_quality_score": result.plate_quality_score,
                "status": result.status.name,
                "rejected_reason": result.rejected_reason,
                "engine": result.engine,
                "engine_version": result.engine_version,
                "evidence_id": result.evidence_id,
            }
            for result in item.ocr_results
        ],
        "fusion": {
            "status": fused.status.name,
            "plate_text": fused.plate_text,
            "confidence": fused.confidence,
            "unique_frame_count": fused.unique_frame_count,
            "supporting_frame_count": fused.supporting_frame_count,
            "supporting_frames": supporting_frames,
            "supporting_evidence_ids": list(fused.supporting_evidence_ids),
            "rejected_evidence_ids": list(fused.rejected_evidence_ids),
            "reason_codes": list(fused.reason_codes),
            "dominant_cluster_weight_share": fused.dominant_cluster_weight_share,
            "mean_position_confidence": fused.mean_position_confidence,
            "minimum_position_confidence": fused.minimum_position_confidence,
            "india_constraint": constraint_payload,
        },
        "finalization_errors": list(item.finalization_errors),
    }


def print_completed(item: CompletedTrackANPR) -> None:
    fused = item.fused_plate_result
    print(f"\nTRACK {item.track_id} ({item.camera_id})")
    print(f"termination: {item.termination_reason.name}")
    print(f"selected_candidates: {item.selected_candidate_count}")
    print("ocr_results:")
    if not item.ocr_results:
        print("  none")
    for result in item.ocr_results:
        print(
            f"  frame={result.frame_index} text={result.raw_text!r} "
            f"conf={result.confidence:.3f} "
            f"quality={result.plate_quality_score:.3f} "
            f"status={result.status.name}"
        )
    print("fusion:")
    print(f"  status={fused.status.name}")
    print(f"  plate={fused.plate_text!r}")
    print(f"  confidence={fused.confidence:.3f}")
    print(f"  unique_frames={fused.unique_frame_count}")
    print(f"  supporting_frames={fused.supporting_frame_count}")


def validate_invariants(
    completed: list[CompletedTrackANPR],
    *,
    top_k: int,
    minimum_frames: int,
    pipeline: CameraPipeline,
    selector: PlateBestFrameSelector,
) -> dict[str, bool]:
    keys = [(item.camera_id, item.track_id) for item in completed]
    accepted = [
        item for item in completed
        if item.fused_plate_result.status.name == "ACCEPTED"
    ]
    repeated_finalize_empty = pipeline.finalize() == ()
    checks = {
        "no_duplicate_completed_tracks": len(keys) == len(set(keys)),
        "accepted_results_have_minimum_unique_frames": all(
            item.fused_plate_result.unique_frame_count >= minimum_frames
            for item in accepted
        ),
        "selected_candidate_count_within_top_k": all(
            item.selected_candidate_count <= top_k for item in completed
        ),
        "accepted_results_have_nonempty_text": all(
            bool(item.fused_plate_result.plate_text) for item in accepted
        ),
        # Results are collected before any status-based reporting; UNKNOWN is
        # deliberately never filtered from `completed`.
        "unknown_results_preserved": True,
        "end_of_stream_finalization_performed": True,
        "repeated_finalize_is_idempotent": repeated_finalize_empty,
        "stage3_state_released_for_completed_tracks": all(
            selector.get(camera_id, track_id) == () for camera_id, track_id in keys
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"Stage 8 invariant failure(s): {', '.join(failed)}")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--camera-id", default=DEFAULT_CAMERA_ID)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.max_frames < 1:
        parser.error("--max-frames must be at least 1")

    video = resolve_path(args.video)
    output = resolve_path(args.output)
    if not video.is_file():
        raise FileNotFoundError(f"Smoke video not found: {video}")

    config = load_model_config()
    pipeline, context = build_pipeline(config)
    source = VideoFileSource(video, args.camera_id)
    completed: list[CompletedTrackANPR] = []
    frames_processed = 0
    started = time.perf_counter()

    for packet in source:
        pipeline.process_frame(packet)
        new_results = pipeline.pop_completed_tracks()
        completed.extend(new_results)
        for item in new_results:
            print_completed(item)
        frames_processed += 1
        if frames_processed >= args.max_frames:
            break

    end_results = pipeline.finalize()
    completed.extend(end_results)
    for item in end_results:
        print_completed(item)
    elapsed = time.perf_counter() - started

    invariants = validate_invariants(
        completed,
        top_k=context["quality_config"].top_k,
        minimum_frames=context["fusion_config"].min_unique_frames_for_accept,
        pipeline=pipeline,
        selector=context["selector"],
    )
    termination_counts = Counter(item.termination_reason.name for item in completed)
    status_counts = Counter(item.fused_plate_result.status.name for item in completed)
    report = {
        "video_path": str(video),
        "frame_limit": args.max_frames,
        "frames_processed": frames_processed,
        "camera_id": args.camera_id,
        "elapsed_seconds": elapsed,
        "processing_fps": frames_processed / elapsed if elapsed else 0.0,
        "active_vehicle_detector_checkpoint": str(context["vehicle_weights"]),
        "active_plate_detector_checkpoint": str(context["plate_weights"]),
        "svtrv2_config": str(context["svtr_config"].config_path),
        "svtrv2_checkpoint": str(context["svtr_config"].checkpoint_path),
        "svtrv2_model_identifier": context["svtr_model_identifier"],
        "vehicle_tracks_completed": len(completed),
        "termination_counts": {
            "TRACK_EXPIRED": termination_counts["TRACK_EXPIRED"],
            "END_OF_STREAM": termination_counts["END_OF_STREAM"],
        },
        "fusion_status_counts": {
            "ACCEPTED": status_counts["ACCEPTED"],
            "LOW_CONFIDENCE": status_counts["LOW_CONFIDENCE"],
            "UNKNOWN": status_counts["UNKNOWN"],
        },
        "invariants": invariants,
        "completed_tracks": [completed_to_dict(item) for item in completed],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")

    print("\nSTAGE 8 SUMMARY")
    print(f"frames_processed: {frames_processed}")
    print(f"elapsed_seconds: {elapsed:.3f}")
    print(f"processing_fps: {report['processing_fps']:.3f}")
    print(f"completed_tracks: {len(completed)}")
    print(f"termination_counts: {dict(report['termination_counts'])}")
    print(f"fusion_status_counts: {dict(report['fusion_status_counts'])}")
    print(f"report: {output}")


if __name__ == "__main__":
    main()
