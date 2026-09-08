from pathlib import Path

from cv.detection.ultralytics_detector import UltralyticsVehicleDetector
from cv.ingestion.video import VideoFileSource

VIDEO_PATH = Path("cv/data/video/traffic.mp4")
WEIGHTS = Path("yolo26n.pt")


def main() -> None:
    detector = UltralyticsVehicleDetector(
        weights=WEIGHTS,
        confidence_threshold=0.25,
        device="cuda",
    )

    source = VideoFileSource(
        path=VIDEO_PATH,
        camera_id="CAM-001",
    )

    for frame_number, frame in enumerate(source):
        detections = detector.detect(frame)

        print(
            f"Frame {frame.frame_index}: "
            f"{len(detections)} vehicles"
        )

        for detection in detections:
            print(
                f"  {detection.class_name:<12} "
                f"{detection.confidence:.3f} "
                f"bbox={detection.bbox}"
            )

        if frame_number >= 9:
            break


if __name__ == "__main__":
    main()