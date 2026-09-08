import logging
import os
import queue
import threading
from dataclasses import dataclass

from backend.database.repository import (
    SQLAlchemyCameraRepository,
    SQLAlchemySightingRepository,
)
from backend.database.session import (
    configured_database_url,
    create_session_factory,
    probe_database,
)
from backend.domain.cameras import CameraNode
from backend.trajectory.conversion import confirmed_sighting_from_anpr
from backend.trajectory.demo import DEMO_CAMERAS
from backend.trajectory.repository import InMemorySightingRepository

log = logging.getLogger(__name__)


class InMemoryCameraRepository:
    def __init__(self): self._items = {}
    def upsert(self, camera): self._items[camera.camera_id] = camera
    def get(self, camera_id): return self._items.get(camera_id)
    def list(self): return tuple(self._items[key] for key in sorted(self._items))


class AcceptedSightingIngestor:
    """Bounded, non-blocking canonical CompletedTrackANPR writer."""
    def __init__(self, repository, cameras, max_pending: int = 64) -> None:
        self.repository, self.cameras = repository, cameras
        self._queue = queue.Queue(maxsize=max_pending)
        self._stop = threading.Event()
        self.last_error: str | None = None
        self._thread = threading.Thread(target=self._run, name="etris-sighting-writer", daemon=True)
        self._thread.start()
    def submit(self, completed) -> bool:
        try: self._queue.put_nowait(completed); return True
        except queue.Full:
            self.last_error = "persistence queue full"; return False
    def _run(self):
        while not self._stop.is_set():
            try: completed = self._queue.get(timeout=.25)
            except queue.Empty: continue
            try:
                camera = self.cameras.get(completed.camera_id)
                if camera:
                    sighting = confirmed_sighting_from_anpr(completed, camera)
                    if sighting: self.repository.add(sighting)
                self.last_error = None
            except Exception as exc:  # noqa: BLE001 - auxiliary persistence must not stop ANPR
                self.last_error = f"{type(exc).__name__}: {exc}"
                log.warning("Sighting persistence failed: %s", self.last_error)
            finally: self._queue.task_done()
    def stop(self):
        self._stop.set(); self._thread.join(timeout=2)


@dataclass
class StorageRuntime:
    sightings: object
    cameras: object
    mode: str
    configured: bool
    connected: bool
    ingestor: AcceptedSightingIngestor
    engine: object | None = None
    session_factory: object | None = None

    def status(self):
        return {"mode": self.mode, "configured": self.configured,
                "connected": self.connected and self.ingestor.last_error is None,
                "last_error": self.ingestor.last_error}


def build_storage_runtime() -> StorageRuntime:
    url = configured_database_url()
    engine = None
    if url:
        try:
            engine, sessions = create_session_factory(url)
            if probe_database(engine):
                sightings, cameras = SQLAlchemySightingRepository(sessions), SQLAlchemyCameraRepository(sessions)
                mode, connected = "postgresql", True
            else: raise ConnectionError("database probe failed")
        except Exception as exc:  # noqa: BLE001 - database fallback is intentionally broad
            log.warning("PostgreSQL unavailable; using memory fallback: %s", exc)
            sightings, cameras, mode, connected = InMemorySightingRepository(), InMemoryCameraRepository(), "postgresql", False
    else:
        sightings, cameras, mode, connected = InMemorySightingRepository(), InMemoryCameraRepository(), "memory", True
    live = CameraNode(
        os.getenv("ETRIS_ANPR_CAMERA_ID", "CAM-DEMO-01"), "Synthetic Demo Video Camera",
        float(os.getenv("ETRIS_ANPR_CAMERA_LATITUDE", "12.9716")),
        float(os.getenv("ETRIS_ANPR_CAMERA_LONGITUDE", "77.5946")),
        "Demo Video Source", "DEMO", None,
    )
    try:
        for camera in DEMO_CAMERAS.values():
            cameras.upsert(camera)
        cameras.upsert(live)
    except Exception as exc:  # noqa: BLE001 - missing/mismatched schema falls back safely
        # A reachable server without migrated tables must not take ANPR down.
        log.warning("PostgreSQL schema unavailable; using memory fallback: %s", exc)
        if engine is not None:
            engine.dispose()
        engine = None
        sightings, cameras, connected = InMemorySightingRepository(), InMemoryCameraRepository(), False
        for camera in DEMO_CAMERAS.values():
            cameras.upsert(camera)
        cameras.upsert(live)
    ingestor = AcceptedSightingIngestor(sightings, cameras)
    return StorageRuntime(sightings, cameras, mode, bool(url), connected, ingestor, engine,
        sessions if connected and engine is not None else None)
