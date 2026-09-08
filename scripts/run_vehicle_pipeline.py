import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg

from cv.detection.ultralytics_detector import UltralyticsVehicleDetector
from cv.ingestion.video import VideoFileSource
from cv.pipeline.camera import CameraPipeline
from cv.tracking.simple import SimpleVehicleTracker

VIDEO_PATH = Path("cv/data/video/traffic.mp4")
WEIGHTS = Path("yolo26n.pt")
TEMP_OUTPUT_PATH = Path("cv/data/video/tracked_output_temp.avi")
OUTPUT_PATH = Path("cv/data/video/tracked_output.mp4")

CAMERA_ID = "CAM-001"

CONFIDENCE_THRESHOLD = 0.25
DEVICE = "cuda"

MAX_FRAMES = None


def main() -> None:
    """Run vehicle detection and tracking over the input video."""

    detector = UltralyticsVehicleDetector(
        weights=WEIGHTS,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        device=DEVICE,
    )

    tracker = SimpleVehicleTracker(
        minimum_iou=0.3,
        max_missed_frames=30,
    )

    pipeline = CameraPipeline(
        detector=detector,
        tracker=tracker,
    )

    source = VideoFileSource(
        path=VIDEO_PATH,
        camera_id=CAMERA_ID,
    )

    capture = cv2.VideoCapture(str(VIDEO_PATH))

    if not capture.isOpened():
        raise RuntimeError(
            f"Unable to open video for metadata: {VIDEO_PATH}"
        )

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    capture.release()

    if fps <= 0:
        raise RuntimeError("Invalid source FPS.")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    writer = cv2.VideoWriter(
    str(TEMP_OUTPUT_PATH),
    cv2.VideoWriter_fourcc(*"MJPG"),
    fps,
    (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Unable to create output video: {OUTPUT_PATH}"
        )

    processed_frames = 0

    try:
        for frame in source:
            tracks = pipeline.process_frame(frame)

            annotated = frame.image.copy()

            for track in tracks:
                x1 = int(track.bbox.x1)
                y1 = int(track.bbox.y1)
                x2 = int(track.bbox.x2)
                y2 = int(track.bbox.y2)

                cv2.rectangle(
                    annotated,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                label = (
                    f"ID {track.track_id} | "
                    f"{track.class_name} | "
                    f"{track.detection_confidence:.2f}"
                )

                text_y = max(20, y1 - 8)

                cv2.putText(
                    annotated,
                    label,
                    (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

            writer.write(annotated)

            processed_frames += 1

            if processed_frames % 30 == 0:
                print(
                    f"Processed {processed_frames} frames | "
                    f"active tracks: {len(tracks)}"
                )

            if (
                MAX_FRAMES is not None
                and processed_frames >= MAX_FRAMES
            ):
                break

    finally:
        writer.release()


    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    subprocess.run(
    [
            ffmpeg,
            "-y",
            "-i",
            str(TEMP_OUTPUT_PATH),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(OUTPUT_PATH),
    ],
       check=True,
)

    TEMP_OUTPUT_PATH.unlink(missing_ok=True) 

    print()
    print("=" * 60)
    print("ETRIS VEHICLE PIPELINE")
    print("=" * 60)
    print(f"Input:            {VIDEO_PATH}")
    print(f"Output:           {OUTPUT_PATH}")
    print(f"Frames processed: {processed_frames}")
    print(f"Resolution:       {width}x{height}")
    print(f"Source FPS:       {fps:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()