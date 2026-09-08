"""Validate Stage 12D with explicitly synthetic Stage 10 demo analytics."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.alerts.models import AlertType
from backend.alerts.repository import InMemoryAlertRepository
from backend.alerts.service import AlertService, CongestionRuntimeIngestor
from backend.analytics.demo import (
    DEMO_WINDOW_END,
    DEMO_WINDOW_START,
    build_demo_analytics,
)
from backend.analytics.models import CongestionLevel, CongestionResult


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ROOT / "runs/stage12d_congestion"))
    return parser.parse_args()


def main():
    args = options()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    repository = InMemoryAlertRepository()
    runtime = CongestionRuntimeIngestor(AlertService(repository))
    snapshots = build_demo_analytics().congestion(DEMO_WINDOW_START, DEMO_WINDOW_END)
    lifecycle = []

    for index in range(2):
        emitted = runtime.process_once(("STAGE10_DEMO_SYNTHETIC", index), snapshots,
            now=DEMO_WINDOW_END.timestamp() + index)
        lifecycle.extend(alert.to_dict() for alert in emitted)

    severe_segments = [item for item in snapshots if item.classification is CongestionLevel.SEVERE]
    for index in range(3):
        recovery = tuple(CongestionResult(item.source_camera, item.target_camera, 3,
            item.baseline_travel_time_s, item.baseline_travel_time_s, 1.0,
            CongestionLevel.FREE_FLOW) for item in severe_segments)
        emitted = runtime.process_once(("STAGE10_DEMO_SYNTHETIC_RECOVERY", index), recovery,
            now=DEMO_WINDOW_END.timestamp() + 10 + index)
        lifecycle.extend(alert.to_dict() for alert in emitted)

    alerts = [alert.to_dict() for alert in repository.list()]
    (output / "congestion_alerts.json").write_text(json.dumps(alerts, indent=2), encoding="utf-8")
    (output / "congestion_lifecycle.json").write_text(json.dumps({
        "data_source": "STAGE10_DEMO_SYNTHETIC",
        "events": lifecycle,
    }, indent=2), encoding="utf-8")
    print(f"alerts: {len(alerts)}")
    print(f"severe_delay: {sum(row['alert_type'] == AlertType.SEVERE_DELAY.value for row in alerts)}")
    print(f"output: {output}")


if __name__ == "__main__":
    main()
