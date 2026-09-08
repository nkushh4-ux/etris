from datetime import datetime

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from backend.database.models import CameraRecord, VehicleSightingRecord
from backend.domain.cameras import CameraNode
from backend.domain.sightings import SightingStatus, VehicleSighting
from backend.trajectory.repository import SightingRepository


def _sighting(row: VehicleSightingRecord) -> VehicleSighting:
    return VehicleSighting(row.sighting_id, row.plate_text, row.camera_id, row.observed_at,
        row.latitude, row.longitude, row.plate_confidence, row.source_track_id,
        row.direction, SightingStatus(row.status), row.source_type or "ANPR_FUSION")


class SQLAlchemySightingRepository(SightingRepository):
    def __init__(self, session_factory) -> None: self.session_factory = session_factory

    def add(self, sighting: VehicleSighting) -> bool:
        row = VehicleSightingRecord(
            sighting_id=sighting.sighting_id, plate_text=sighting.plate_text,
            camera_id=sighting.camera_id, observed_at=sighting.timestamp,
            plate_confidence=sighting.plate_confidence, source_track_id=sighting.source_track_id,
            direction=sighting.direction, status=sighting.status.value, frame_id=None,
            source_type=sighting.source, latitude=sighting.latitude, longitude=sighting.longitude)
        with self.session_factory() as session:
            session.add(row)
            try: session.commit(); return True
            except IntegrityError: session.rollback(); return False

    def _query(self, statement):
        statement = statement.where(VehicleSightingRecord.status == SightingStatus.CONFIRMED.value)
        statement = statement.order_by(VehicleSightingRecord.observed_at,
            VehicleSightingRecord.camera_id, VehicleSightingRecord.sighting_id)
        with self.session_factory() as session: return tuple(_sighting(x) for x in session.scalars(statement))

    def get_by_plate(self, plate_text: str):
        return self._query(select(VehicleSightingRecord).where(VehicleSightingRecord.plate_text == plate_text))
    def get_between(self, start: datetime, end: datetime):
        if end < start: raise ValueError("end must not precede start")
        return self._query(select(VehicleSightingRecord).where(VehicleSightingRecord.observed_at.between(start, end)))
    def get_by_camera(self, camera_id: str):
        return self._query(select(VehicleSightingRecord).where(VehicleSightingRecord.camera_id == camera_id))
    def get_recent(self, limit: int = 50, camera_id: str | None = None):
        statement = select(VehicleSightingRecord)
        if camera_id: statement = statement.where(VehicleSightingRecord.camera_id == camera_id)
        rows = self._query(statement)
        return rows[-limit:]


class SQLAlchemyCameraRepository:
    def __init__(self, session_factory) -> None: self.session_factory = session_factory
    def upsert(self, camera: CameraNode) -> None:
        with self.session_factory() as session:
            row = session.get(CameraRecord, camera.camera_id) or CameraRecord(camera_id=camera.camera_id)
            for name in ("name", "latitude", "longitude", "road_name", "sector", "direction", "is_active"):
                setattr(row, name, getattr(camera, name))
            session.add(row); session.commit()
    def get(self, camera_id: str) -> CameraNode | None:
        with self.session_factory() as session:
            row = session.get(CameraRecord, camera_id)
            return self._domain(row) if row else None
    def list(self) -> tuple[CameraNode, ...]:
        with self.session_factory() as session:
            return tuple(self._domain(x) for x in session.scalars(select(CameraRecord).order_by(CameraRecord.camera_id)))
    @staticmethod
    def _domain(row):
        return CameraNode(row.camera_id,row.name,row.latitude,row.longitude,row.road_name,row.sector,row.direction,row.is_active)
