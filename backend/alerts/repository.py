from abc import ABC, abstractmethod
from threading import RLock

from backend.alerts.models import Alert, AlertStatus, AlertType


class AlertRepository(ABC):
    @abstractmethod
    def save(self,alert:Alert)->None: ...
    @abstractmethod
    def get(self,alert_id:str)->Alert|None: ...
    @abstractmethod
    def list(self,*,alert_type:AlertType|None=None,status:AlertStatus|None=None,camera_id:str|None=None,zone_id:str|None=None)->tuple[Alert,...]: ...
    @abstractmethod
    def get_active_by_key(self,key:tuple)->Alert|None: ...


class InMemoryAlertRepository(AlertRepository):
    def __init__(self): self._items={}; self._active={}; self._lock=RLock()
    def save(self,alert):
        with self._lock:
            self._items[alert.alert_id]=alert
            if alert.status is AlertStatus.CLEARED: self._active.pop(alert.lifecycle_key,None)
            else: self._active[alert.lifecycle_key]=alert.alert_id
    def get(self,alert_id):
        with self._lock: return self._items.get(alert_id)
    def get_active_by_key(self,key):
        with self._lock:
            identifier=self._active.get(key); return self._items.get(identifier) if identifier else None
    def list(self,*,alert_type=None,status=None,camera_id=None,zone_id=None):
        with self._lock: rows=tuple(self._items.values())
        return tuple(sorted((x for x in rows if (alert_type is None or x.alert_type is alert_type)
            and (status is None or x.status is status) and (camera_id is None or x.camera_id==camera_id)
            and (zone_id is None or x.zone_id==zone_id)),key=lambda x:(x.started_at,x.alert_id)))
