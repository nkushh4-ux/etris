"""Run one real crop through the production SVTRv2 adapter."""

from datetime import UTC, datetime

import cv2

from cv.detection.models import BoundingBox
from cv.ocr.svtrv2 import PROJECT_ROOT, SVTRv2Config, SVTRv2Engine
from cv.plate.models import PlateCandidate, PlateQualityMetrics


def main() -> None:
    crop_path = (
        PROJECT_ROOT
        / "data/ocr_benchmark/development_video/crops/DEV-001.jpg"
    )
    crop = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
    if crop is None:
        raise FileNotFoundError(f"Could not read smoke crop: {crop_path}")

    height, width = crop.shape[:2]
    metrics = PlateQualityMetrics(
        sharpness=0.0,
        exposure=0.0,
        contrast=0.0,
        size=0.0,
        detector_confidence=0.0,
        clipping=1.0,
        laplacian_variance=0.0,
        mean_intensity=0.0,
        intensity_stddev=0.0,
        crop_width_px=width,
        crop_height_px=height,
        visible_fraction=1.0,
    )
    candidate = PlateCandidate(
        camera_id="development-smoke",
        track_id=1,
        frame_index=1,
        timestamp=datetime.now(UTC),
        bbox=BoundingBox(0, 0, width, height),
        detector_confidence=0.0,
        metrics=metrics,
        quality_score=0.0,
        crop=crop,
    )
    model_dir = (
        PROJECT_ROOT
        / "external/OpenOCR/weights/svtrv2_s/drive-download-20260904T165653Z-1-001"
    )
    engine = SVTRv2Engine(SVTRv2Config(
        openocr_root=PROJECT_ROOT / "external/OpenOCR",
        config_path=model_dir / "config_infer.yml",
        checkpoint_path=model_dir / "best.pth",
        device="auto",
    ))
    result = engine.recognize(candidate)
    print(f"status={result.status.value}")
    print(f"raw_text={result.raw_text!r}")
    print(f"normalized_text={result.normalized_text!r}")
    print(f"confidence={result.confidence:.6f}")
    print(f"engine={result.engine}")
    print(f"engine_version={result.engine_version}")
    if result.rejected_reason:
        print(f"rejected_reason={result.rejected_reason}")


if __name__ == "__main__":
    main()
