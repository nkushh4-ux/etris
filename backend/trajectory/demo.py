"""Deterministic DEMO/SYNTHETIC Stage 9 data; not deployed-camera evidence."""

from datetime import UTC, datetime, timedelta

from backend.domain.cameras import CameraNode
from backend.domain.sightings import VehicleSighting
from backend.trajectory.models import CameraEdge
from backend.trajectory.reconstruction import CameraTopology, TrajectoryReconstructor
from backend.trajectory.repository import InMemorySightingRepository
from backend.trajectory.service import TrajectoryService

DEMO_CAMERAS = {
    item.camera_id: item for item in (
        CameraNode("CAM-01", "Demo North Gate", 12.9716, 77.5946, "Demo Ring Road", "North", "SOUTH"),
        CameraNode("CAM-02", "Demo East Junction", 12.9730, 77.6070, "Demo East Road", "East", "EAST"),
        CameraNode("CAM-03", "Demo Central Junction", 12.9650, 77.6060, "Demo Central Road", "Central", "SOUTH"),
        CameraNode("CAM-04", "Demo West Junction", 12.9620, 77.5880, "Demo West Road", "West", "WEST"),
        CameraNode("CAM-05", "Demo South Gate", 12.9510, 77.6120, "Demo South Road", "South", "SOUTH"),
    )
}

DEMO_EDGES = (
    CameraEdge("CAM-01", "CAM-03", 1800, 90, 600),
    CameraEdge("CAM-03", "CAM-05", 2400, 120, 720),
    CameraEdge("CAM-01", "CAM-02", 1500, 75, 480),
    CameraEdge("CAM-02", "CAM-03", 1200, 60, 420),
    CameraEdge("CAM-03", "CAM-04", 2100, 100, 600),
)


def build_demo_service() -> TrajectoryService:
    repository = InMemorySightingRepository()
    base = datetime(2026, 9, 7, 10, 2, 14, tzinfo=UTC)
    rows = (
        ("KA02MN1826", "CAM-01", base, .94, 101),
        ("KA02MN1826", "CAM-03", base + timedelta(minutes=4, seconds=28), .92, 203),
        ("KA02MN1826", "CAM-05", base + timedelta(minutes=9, seconds=5), .95, 305),
        ("KA01AB1234", "CAM-01", base + timedelta(seconds=30), .90, 111),
        ("KA01AB1234", "CAM-02", base + timedelta(minutes=4), .88, 212),
        ("KA05XY7788", "CAM-03", base + timedelta(minutes=1), .91, 221),
        ("KA05XY7788", "CAM-04", base + timedelta(minutes=6), .89, 414),
    )
    for index, (plate, camera_id, timestamp, confidence, track_id) in enumerate(rows, 1):
        camera = DEMO_CAMERAS[camera_id]
        repository.add(VehicleSighting(
            f"demo-{index:03d}", plate, camera_id, timestamp,
            camera.latitude, camera.longitude, confidence, track_id,
            camera.direction, source="DEMO_SYNTHETIC",
        ))
    reconstructor = TrajectoryReconstructor(repository, DEMO_CAMERAS, CameraTopology(DEMO_EDGES))
    return TrajectoryService(repository, reconstructor)

