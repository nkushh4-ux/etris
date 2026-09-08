"""Run the locked production ANPR pipeline with live annotations."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.streaming.anpr_stream import ANPRStreamService
from backend.streaming.factory import (
    DEFAULT_OUTPUT,
    DEFAULT_VIDEO,
    production_pipeline_factory,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--display", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-output", nargs="?", const=DEFAULT_OUTPUT, type=Path)
    parser.add_argument("--events-output", type=Path)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--camera-id", default="CAM-DEMO-01")
    args = parser.parse_args()
    service = ANPRStreamService(args.video, args.camera_id, production_pipeline_factory,
                                loop=args.loop, max_frames=args.max_frames)
    service.run(display=args.display, save_output=args.save_output)
    status = service.status()
    print(
        f"frames={status.frame + 1} source_fps={status.source_fps:.3f} "
        f"playback_fps={status.playback_fps:.3f} inference_fps={status.inference_fps:.3f} "
        f"status={status.pipeline_status}"
    )
    print(f"preview_metrics={service.preview_metrics()}")
    print(f"performance_counts={service.performance_counts()}")
    print(f"india_constraint_diagnostics={json.dumps(service.constraint_diagnostics(), indent=2)}")
    print(f"stage_timings={json.dumps(service.timing_report(), indent=2)}")
    for event in reversed(service.recent_events()):
        print(event.to_dict())
    if args.save_output:
        events_output = args.events_output or args.save_output.with_name(
            f"{args.save_output.stem}_events.json"
        )
        events_output.parent.mkdir(parents=True, exist_ok=True)
        events_output.write_text(
            json.dumps([event.to_dict() for event in reversed(service.recent_events())], indent=2),
            encoding="utf-8",
        )
        print(f"events={events_output}")


if __name__ == "__main__":
    main()
