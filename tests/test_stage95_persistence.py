from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend.api.app import create_app
from backend.database.base import Base
from backend.database.models import CameraRecord, VehicleSightingRecord
from backend.database.repository import (
    SQLAlchemyCameraRepository,
    SQLAlchemySightingRepository,
)
from backend.database.runtime import (
    AcceptedSightingIngestor,
    InMemoryCameraRepository,
    build_storage_runtime,
)
from backend.domain.cameras import CameraNode
from backend.domain.sightings import SightingStatus, VehicleSighting
from cv.ocr.models import FusedPlateResult, FusionStatus
from cv.pipeline.models import CompletedTrackANPR, TrackTerminationReason

BASE = datetime(2026, 9, 7, 10, 0, tzinfo=UTC)
CAMERA = CameraNode("CAM-DB", "Database Camera", 12.1, 77.2, direction="NORTH")


def repositories():
    engine = create_engine("sqlite+pysqlite:///:memory:", connect_args={"check_same_thread": False},
                           poolclass=StaticPool)
    Base.metadata.create_all(engine)
    sessions = sessionmaker(engine, expire_on_commit=False)
    cameras = SQLAlchemyCameraRepository(sessions)
    cameras.upsert(CAMERA)
    return engine, cameras, SQLAlchemySightingRepository(sessions)


def sighting(identifier="s-1", seconds=0, camera=CAMERA, status=SightingStatus.CONFIRMED):
    return VehicleSighting(identifier, "KA02MN1826", camera.camera_id,
                           BASE + timedelta(seconds=seconds), camera.latitude, camera.longitude,
                           .93, 19, camera.direction, status)


def completed(status=FusionStatus.ACCEPTED):
    fused = FusedPlateResult(status, "KA02MN1826", .93, CAMERA.camera_id, 19,
                            3, 3, 3, 3, 1.0, 1.0, .9, .8, (), (), (), (), ())
    return CompletedTrackANPR(CAMERA.camera_id, 19, TrackTerminationReason.TRACK_EXPIRED,
                              299, BASE, 3, 3, (), fused)


def test_database_models_expose_required_fields():
    assert {"camera_id", "latitude", "longitude", "is_active"} <= set(CameraRecord.__table__.columns.keys())
    assert {"sighting_id", "plate_text", "camera_id", "observed_at", "plate_confidence",
            "source_track_id", "status", "frame_id", "source_type"} <= set(VehicleSightingRecord.__table__.columns.keys())


def test_sql_repository_idempotency_order_camera_and_time_queries():
    engine, cameras, repository = repositories()
    other = CameraNode("CAM-OTHER", "Other", 12.2, 77.3)
    cameras.upsert(other)
    later, earlier, isolated = sighting("s-2", 20), sighting("s-1", 10), sighting("s-3", 15, other)
    assert repository.add(later) and repository.add(earlier) and repository.add(isolated)
    assert not repository.add(earlier)
    assert [x.sighting_id for x in repository.get_by_plate("KA02MN1826")] == ["s-1", "s-3", "s-2"]
    assert [x.sighting_id for x in repository.get_by_camera(CAMERA.camera_id)] == ["s-1", "s-2"]
    assert [x.sighting_id for x in repository.get_between(BASE + timedelta(seconds=12), BASE + timedelta(seconds=18))] == ["s-3"]
    assert [x.sighting_id for x in repository.get_recent(2)] == ["s-3", "s-2"]
    engine.dispose()


def test_ingestor_persists_only_accepted_completed_tracks():
    engine, cameras, repository = repositories()
    ingestor = AcceptedSightingIngestor(repository, cameras, max_pending=2)
    assert ingestor.submit(completed(FusionStatus.UNKNOWN))
    assert ingestor.submit(completed(FusionStatus.ACCEPTED))
    ingestor._queue.join()
    assert [x.plate_text for x in repository.get_by_plate("KA02MN1826")] == ["KA02MN1826"]
    ingestor.stop(); engine.dispose()


def test_ingestor_database_failure_isolated_from_canonical_processing():
    class BrokenRepository:
        def add(self, _): raise RuntimeError("database down")
    cameras = InMemoryCameraRepository(); cameras.upsert(CAMERA)
    ingestor = AcceptedSightingIngestor(BrokenRepository(), cameras)
    assert ingestor.submit(completed())
    ingestor._queue.join()
    assert "database down" in (ingestor.last_error or "")
    ingestor.stop()


def test_no_database_url_uses_memory_fallback(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    runtime = build_storage_runtime()
    assert runtime.status() == {"mode": "memory", "configured": False, "connected": True, "last_error": None}
    runtime.ingestor.stop()


class FakeStream:
    def start(self): pass
    def stop(self): pass
    def status(self):
        from backend.streaming.anpr_stream import ANPRStreamStatus
        return ANPRStreamStatus(False, 0, 0, "demo.mp4", "CAM-DB", "IDLE")


def test_api_status_and_stage9_serialization(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    app = create_app(stream_service=FakeStream())
    with TestClient(app) as client:
        status = client.get("/api/anpr/status?mode=LIVE_ANALYSIS")
        assert status.status_code == 200
        assert status.json()["backend_online"] is True
        assert status.json()["database"]["mode"] == "memory"
        cameras = client.get("/api/cameras")
        assert cameras.status_code == 200 and cameras.json()[0]["camera_id"]
        assert client.get("/api/sightings/recent").json() == []
        trajectory = client.get("/api/trajectory/UNKNOWN").json()
        assert trajectory["plate"] == "UNKNOWN" and trajectory["points"] == []
