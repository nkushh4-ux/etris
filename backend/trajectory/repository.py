from abc import ABC, abstractmethod
from datetime import datetime

from backend.domain.sightings import SightingStatus, VehicleSighting


def _order(item: VehicleSighting) -> tuple[datetime, str, str]:
    return item.timestamp, item.camera_id, item.sighting_id


class SightingRepository(ABC):
    @abstractmethod
    def add(self, sighting: VehicleSighting) -> bool: ...

    @abstractmethod
    def get_by_plate(self, plate_text: str) -> tuple[VehicleSighting, ...]: ...

    @abstractmethod
    def get_between(self, start: datetime, end: datetime) -> tuple[VehicleSighting, ...]: ...

    @abstractmethod
    def get_by_camera(self, camera_id: str) -> tuple[VehicleSighting, ...]: ...

    def get_recent(self, limit: int = 50, camera_id: str | None = None) -> tuple[VehicleSighting, ...]:
        rows = self.get_by_camera(camera_id) if camera_id else self.get_between(
            datetime.min.replace(tzinfo=None), datetime.max.replace(tzinfo=None)
        )
        return rows[-limit:]


class InMemorySightingRepository(SightingRepository):
    def __init__(self) -> None:
        self._by_id: dict[str, VehicleSighting] = {}
        self._duplicate_keys: set[tuple] = set()

    def add(self, sighting: VehicleSighting) -> bool:
        existing = self._by_id.get(sighting.sighting_id)
        if existing is not None:
            if existing == sighting:
                return False
            raise ValueError(f"sighting_id already exists: {sighting.sighting_id}")
        if sighting.duplicate_key in self._duplicate_keys:
            return False
        self._by_id[sighting.sighting_id] = sighting
        self._duplicate_keys.add(sighting.duplicate_key)
        return True

    def _confirmed(self):
        return (item for item in self._by_id.values() if item.status is SightingStatus.CONFIRMED)

    def get_by_plate(self, plate_text: str) -> tuple[VehicleSighting, ...]:
        return tuple(sorted((x for x in self._confirmed() if x.plate_text == plate_text), key=_order))

    def get_between(self, start: datetime, end: datetime) -> tuple[VehicleSighting, ...]:
        if end < start:
            raise ValueError("end must not precede start")
        return tuple(sorted((x for x in self._confirmed() if start <= x.timestamp <= end), key=_order))

    def get_by_camera(self, camera_id: str) -> tuple[VehicleSighting, ...]:
        return tuple(sorted((x for x in self._confirmed() if x.camera_id == camera_id), key=_order))

    def get_recent(self, limit: int = 50, camera_id: str | None = None) -> tuple[VehicleSighting, ...]:
        rows = (x for x in self._confirmed() if camera_id is None or x.camera_id == camera_id)
        return tuple(sorted(rows, key=_order)[-limit:])
