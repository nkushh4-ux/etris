"""Deterministic Stage 10 DEMO/SYNTHETIC data; never loaded into production implicitly."""

from datetime import UTC, datetime, timedelta

from backend.analytics.service import TrafficAnalyticsService
from backend.domain.sightings import VehicleSighting
from backend.trajectory.demo import DEMO_CAMERAS, DEMO_EDGES
from backend.trajectory.repository import InMemorySightingRepository

DEMO_WINDOW_START = datetime(2026, 9, 7, 8, 0, tzinfo=UTC)
DEMO_WINDOW_END = DEMO_WINDOW_START + timedelta(hours=2)


def build_demo_analytics() -> TrafficAnalyticsService:
    """Return an isolated repository containing 30 synthetic multi-camera vehicles."""
    repository = InMemorySightingRepository()
    sequence = 0
    for index in range(30):
        plate = f"KA{index % 9 + 1:02d}ET{index:04d}"
        start = DEMO_WINDOW_START + timedelta(minutes=index * 2)
        if index < 6: route = ("CAM-01", "CAM-03")
        elif index < 18: route = ("CAM-01", "CAM-03", "CAM-05")
        elif index < 24: route = ("CAM-01", "CAM-02", "CAM-03")
        else: route = ("CAM-03", "CAM-04")
        elapsed = 240 if index < 10 and route[:2] == ("CAM-01", "CAM-03") else 105
        timestamp = start
        for step, camera_id in enumerate(route):
            if step:
                previous = route[step - 1]
                edge = next(x for x in DEMO_EDGES if (x.source_camera_id, x.target_camera_id) == (previous, camera_id))
                timestamp += timedelta(seconds=elapsed if step == 1 else max(edge.min_travel_seconds + 30, 150))
            camera = DEMO_CAMERAS[camera_id]; sequence += 1
            repository.add(VehicleSighting(f"stage10-demo-{sequence:03d}", plate, camera_id,
                timestamp, camera.latitude, camera.longitude, .90, index + 1,
                camera.direction, source="STAGE10_DEMO_SYNTHETIC"))
    return TrafficAnalyticsService(repository, DEMO_CAMERAS, DEMO_EDGES)
