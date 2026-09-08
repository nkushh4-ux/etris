from datetime import UTC, datetime
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from backend.alerts.watchlist import WatchlistEntry, WatchlistSeverity

router=APIRouter(prefix="/api/watchlist",tags=["watchlist"])


class WatchlistCreate(BaseModel):
    plate_text:str; reason:str; severity:WatchlistSeverity=WatchlistSeverity.HIGH; source:str
    active:bool=True; expires_at:datetime|None=None; notes:str|None=None


class WatchlistPatch(BaseModel):
    reason:str|None=None; severity:WatchlistSeverity|None=None; source:str|None=None
    active:bool|None=None; expires_at:datetime|None=None; notes:str|None=None


@router.get("")
def list_watchlist(request:Request): return [item.to_dict() for item in request.app.state.watchlist_repository.list()]


@router.post("",status_code=201)
def add_watchlist(payload:WatchlistCreate,request:Request):
    now=datetime.now(UTC)
    try:
        item=WatchlistEntry(f"WL-{uuid4().hex[:12].upper()}",payload.plate_text,payload.reason,payload.severity,
            payload.source,payload.active,now,payload.expires_at,payload.notes)
    except ValueError as error: raise HTTPException(422,str(error)) from error
    request.app.state.watchlist_repository.save(item); return item.to_dict()


@router.patch("/{watchlist_id}")
def patch_watchlist(watchlist_id:str,payload:WatchlistPatch,request:Request):
    current=request.app.state.watchlist_repository.get(watchlist_id)
    if current is None: raise HTTPException(404,"Watchlist entry not found")
    changes=payload.model_dump(exclude_unset=True)
    values={name:getattr(current,name) for name in current.__dataclass_fields__}; values.update(changes)
    try: updated=WatchlistEntry(**values)
    except ValueError as error: raise HTTPException(422,str(error)) from error
    request.app.state.watchlist_repository.save(updated); return updated.to_dict()


@router.delete("/{watchlist_id}")
def deactivate_watchlist(watchlist_id:str,request:Request):
    current=request.app.state.watchlist_repository.get(watchlist_id)
    if current is None: raise HTTPException(404,"Watchlist entry not found")
    values={name:getattr(current,name) for name in current.__dataclass_fields__}; values["active"]=False
    updated=WatchlistEntry(**values); request.app.state.watchlist_repository.save(updated); return updated.to_dict()
