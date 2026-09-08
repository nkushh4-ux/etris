from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum


class WatchlistSeverity(str, Enum):
    LOW="LOW"; MEDIUM="MEDIUM"; HIGH="HIGH"; CRITICAL="CRITICAL"


@dataclass(frozen=True, slots=True)
class WatchlistEntry:
    watchlist_id:str; plate_text:str; reason:str; severity:WatchlistSeverity; source:str
    active:bool; created_at:datetime; expires_at:datetime|None=None; notes:str|None=None

    def __post_init__(self):
        if not self.watchlist_id.strip() or not self.plate_text.strip() or not self.reason.strip() or not self.source.strip():
            raise ValueError("watchlist ID, plate, reason, and source are required")
        if self.created_at.tzinfo is None or self.expires_at is not None and self.expires_at.tzinfo is None:
            raise ValueError("watchlist timestamps must be timezone-aware")

    def is_effective(self,at:datetime|None=None):
        instant=at or datetime.now(UTC)
        return self.active and (self.expires_at is None or self.expires_at>instant)

    def to_dict(self):
        return {"watchlist_id":self.watchlist_id,"plate_text":self.plate_text,"reason":self.reason,
            "severity":self.severity.value,"source":self.source,"active":self.active,
            "created_at":self.created_at.isoformat(),"expires_at":self.expires_at.isoformat() if self.expires_at else None,
            "notes":self.notes,"demo_entry":self.source.upper().startswith("DEMO")}
