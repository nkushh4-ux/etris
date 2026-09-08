import argparse
import time
from collections.abc import Sequence
from pathlib import Path

from cv.detection.detector import VehicleDetector
from cv.ingestion.video import VideoFileSource
from cv.pipeline.camera import CameraPipeline
from cv.tracking.models import TrackState
from cv.tracking.simple import SimpleVehicleTracker


class BenchmarkDetector(VehicleDetector):
    """Placeholder detector used until a real model is configured."""

    def detect(self, frame):
        raise NotImplementedError(
            "A real vehicle detector must be configured "
            "before running the benchmark."
        )


def run_benchmark(
    video_path: str | Path,
    camera_id: str,
    detector: VehicleDetector,
    max_frames: int | None = None,
) -> None:
    """Run detection and tracking over a video."""

    source = VideoFileSource(
        path=video_path,
        camera_id=camera_id,
    )

    tracker = SimpleVehicleTracker()

    pipeline = CameraPipeline(
        detector=detector,
        tracker=tracker,
    )

    processed_frames = 0
    total_tracks = 0

    start_time = time.perf_counter()

    for frame in source:
        tracks: Sequence[TrackState] = pipeline.process_frame(frame)

        processed_frames += 1
        total_tracks += len(tracks)

        if max_frames is not None and processed_frames >= max_frames:
            break

    elapsed = time.perf_counter() - start_time

    if elapsed <= 0:
        raise RuntimeError("Benchmark elapsed time was zero.")

    fps = processed_frames / elapsed

    print()
    print("=" * 60)
    print("ETRIS CV BENCHMARK")
    print("=" * 60)
    print(f"Video:             {Path(video_path)}")
    print(f"Camera:            {camera_id}")
    print(f"Frames processed:  {processed_frames}")
    print(f"Elapsed time:      {elapsed:.3f} s")
    print(f"Throughput:        {fps:.2f} FPS")
    print(f"Total track output:{total_tracks}")
    print("=" * 60)


def build_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser."""

    parser = argparse.ArgumentParser(
        description="Run the ETRIS CV benchmark."
    )

    parser.add_argument(
        "video",
        type=Path,
        help="Path to the input video.",
    )

    parser.add_argument(
        "--camera-id",
        default="CAM-001",
        help="Camera identifier.",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process.",
    )

    return parser


def main() -> None:
    """CLI entry point."""

    parser = build_parser()
    args = parser.parse_args()

    run_benchmark(
        video_path=args.video,
        camera_id=args.camera_id,
        detector=BenchmarkDetector(),
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()