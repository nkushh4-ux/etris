from collections import deque
from threading import RLock

from backend.alerts.congestion import CongestionAlertHandler, CongestionAlertPolicy
from backend.alerts.repository import AlertRepository
from backend.alerts.restricted_parking import RestrictedParkingAlertHandler
from backend.alerts.restricted_zone import RestrictedZoneAlertHandler


class AlertService:
    def __init__(self, repository: AlertRepository,
                 congestion_policy: CongestionAlertPolicy | None = None) -> None:
        self.repository = repository
        self.parking = RestrictedParkingAlertHandler(repository)
        self.restricted_zone = RestrictedZoneAlertHandler(repository)
        self.congestion = CongestionAlertHandler(repository, congestion_policy)

    def process_parking_events(self, events):
        return tuple(alert for event in events if (alert := self.parking.handle(event)) is not None)

    def process_restricted_zone_events(self, events):
        return tuple(alert for event in events if (alert := self.restricted_zone.handle(event)) is not None)

    def process_congestion_snapshots(self, snapshots, *, now: float | None = None):
        """Evaluate Stage 10 CongestionResult snapshots; return emitted alerts."""
        return self.congestion.process_snapshots(tuple(snapshots), now=now)


class CongestionRuntimeIngestor:
    """Idempotently feed distinct Stage 10 snapshot windows to one alert service."""

    def __init__(self, service: AlertService, *, history_size: int = 256) -> None:
        if history_size < 1:
            raise ValueError("history_size must be positive")
        self.service = service
        self._history_size = history_size
        self._keys = set()
        self._order = deque()
        self._lock = RLock()

    def process_once(self, snapshot_key, snapshots, *, now: float | None = None):
        with self._lock:
            if snapshot_key in self._keys:
                return ()
            self._keys.add(snapshot_key)
            self._order.append(snapshot_key)
            if len(self._order) > self._history_size:
                self._keys.remove(self._order.popleft())
            return self.service.process_congestion_snapshots(snapshots, now=now)
