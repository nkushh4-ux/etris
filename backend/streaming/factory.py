from pathlib import Path

from backend.config.loader import load_model_config

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VIDEO = ROOT / "cv/data/video/traffic.mp4"
DEFAULT_OUTPUT = ROOT / "runs/anpr_demo/annotated_traffic.mp4"
DEFAULT_EVENTS = ROOT / "runs/anpr_demo/annotated_traffic_events.json"


class ReusableProductionPipelineFactory:
    """Create fresh temporal state while loading each ML model only once."""

    def __init__(self) -> None:
        self._components = None

    def __call__(self):
        if self._components is None:
            from cv.tracking.types import TrackingAlgorithm
            from scripts.run_anpr_smoke import build_pipeline
            config = load_model_config()
            pipeline, context = build_pipeline(config)
            performance = config.get("live_performance", {})
            intervals = tuple(performance.get("plate_detection_intervals", [1, 1, 1, 1]))
            if len(intervals) != 4:
                raise ValueError("live_performance.plate_detection_intervals must have four values")
            pipeline.vehicle_detection_interval = int(performance.get("vehicle_detection_interval", 1))
            pipeline.plate_detection_intervals = tuple(map(int, intervals))
            pipeline.plate_detector.max_batch_size = int(performance.get("plate_detector_max_batch_size", 4))
            self._components = (
                pipeline.detector,
                pipeline.plate_detector,
                pipeline.ocr_coordinator.engine,
                context["quality_config"],
                context["fusion_config"],
                context["india_constraint_config"],
                TrackingAlgorithm(config["tracking"]["algorithm"]),
                performance,
            )
            return pipeline

        from cv.ocr.coordinator import TrackOCRCoordinator
        from cv.ocr.fusion import MultiFrameOCRFuser
        from cv.pipeline.camera import CameraPipeline
        from cv.plate.quality import PlateQualityAssessor
        from cv.plate.selection import PlateBestFrameSelector
        from cv.tracking.factory import create_tracker
        from cv.tracking.types import TrackingAlgorithm

        detector, plate_detector, engine, quality_config, fusion_config, india_config, tracking_algorithm, performance = self._components
        from cv.ocr.india import IndiaPlateConstraintDecoder
        return CameraPipeline(
            detector=detector,
            tracker=create_tracker(tracking_algorithm),
            plate_detector=plate_detector,
            quality_assessor=PlateQualityAssessor(quality_config),
            best_frame_selector=PlateBestFrameSelector(
                top_k=quality_config.top_k,
                min_frame_gap=quality_config.min_frame_gap,
            ),
            ocr_coordinator=TrackOCRCoordinator(
                engine, MultiFrameOCRFuser(fusion_config, IndiaPlateConstraintDecoder(india_config))
            ),
            vehicle_detection_interval=int(performance.get("vehicle_detection_interval", 1)),
            plate_detection_intervals=tuple(map(int, performance.get("plate_detection_intervals", [1, 1, 1, 1]))),
        )


_production_factory = ReusableProductionPipelineFactory()


def production_pipeline_factory():
    return _production_factory()
