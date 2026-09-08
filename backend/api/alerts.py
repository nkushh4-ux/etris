from fastapi import APIRouter, HTTPException, Request

from backend.alerts.models import AlertStatus, AlertType

router=APIRouter(prefix="/api/alerts",tags=["alerts"])


def _filters(alert_type,status):
    try:
        return (AlertType(alert_type) if alert_type else None,AlertStatus(status) if status else None)
    except ValueError as error: raise HTTPException(422,str(error)) from error


@router.get("")
def alerts(request:Request,alert_type:str|None=None,status:str|None=None,camera_id:str|None=None,zone_id:str|None=None):
    kind,state=_filters(alert_type,status)
    return [x.to_dict() for x in request.app.state.alert_repository.list(alert_type=kind,status=state,camera_id=camera_id,zone_id=zone_id)]


@router.get("/active")
def active(request:Request,alert_type:str|None=None,camera_id:str|None=None,zone_id:str|None=None):
    kind,_=_filters(alert_type,None); rows=request.app.state.alert_repository.list(alert_type=kind,camera_id=camera_id,zone_id=zone_id)
    return [x.to_dict() for x in rows if x.status is not AlertStatus.CLEARED]


@router.get("/{alert_id}")
def by_id(alert_id:str,request:Request):
    item=request.app.state.alert_repository.get(alert_id)
    if item is None: raise HTTPException(404,"Alert not found")
    return item.to_dict()
