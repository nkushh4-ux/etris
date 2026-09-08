from abc import ABC, abstractmethod
from threading import RLock

from sqlalchemy import select

from backend.alerts.watchlist import WatchlistEntry, WatchlistSeverity
from backend.database.models import WatchlistRecord


class WatchlistRepository(ABC):
    @abstractmethod
    def save(self,entry:WatchlistEntry)->None: ...
    @abstractmethod
    def get(self,watchlist_id:str)->WatchlistEntry|None: ...
    @abstractmethod
    def list(self)->tuple[WatchlistEntry,...]: ...
    @abstractmethod
    def find_active_exact(self,plate_text:str,at)->tuple[WatchlistEntry,...]: ...


class InMemoryWatchlistRepository(WatchlistRepository):
    def __init__(self): self._items={}; self._lock=RLock()
    def save(self,entry):
        with self._lock: self._items[entry.watchlist_id]=entry
    def get(self,watchlist_id):
        with self._lock: return self._items.get(watchlist_id)
    def list(self):
        with self._lock: return tuple(self._items[key] for key in sorted(self._items))
    def find_active_exact(self,plate_text,at):
        return tuple(item for item in self.list() if item.plate_text==plate_text and item.is_effective(at))


class SQLAlchemyWatchlistRepository(WatchlistRepository):
    def __init__(self,session_factory): self.session_factory=session_factory
    def save(self,entry):
        with self.session_factory() as session:
            row=session.get(WatchlistRecord,entry.watchlist_id) or WatchlistRecord(watchlist_id=entry.watchlist_id)
            for name in ("plate_text","reason","source","active","created_at","expires_at","notes"):
                setattr(row,name,getattr(entry,name))
            row.severity=entry.severity.value; session.add(row); session.commit()
    def get(self,watchlist_id):
        with self.session_factory() as session: return _domain(session.get(WatchlistRecord,watchlist_id))
    def list(self):
        with self.session_factory() as session:
            return tuple(_domain(row) for row in session.scalars(select(WatchlistRecord).order_by(WatchlistRecord.watchlist_id)))
    def find_active_exact(self,plate_text,at):
        return tuple(item for item in self.list() if item.plate_text==plate_text and item.is_effective(at))


def _domain(row):
    if row is None: return None
    return WatchlistEntry(row.watchlist_id,row.plate_text,row.reason,WatchlistSeverity(row.severity),row.source,
        row.active,row.created_at,row.expires_at,row.notes)
