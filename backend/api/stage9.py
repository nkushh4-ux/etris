from datetime import datetime

from fastapi import APIRouter, Query, Request

from backend.trajectory.demo import DEMO_EDGES
from backend.trajectory.reconstruction import CameraTopology, TrajectoryReconstructor

router = APIRouter(prefix="/api", tags=["stage9"])

def _serialize(item):
    return {"sighting_id":item.sighting_id,"plate_text":item.plate_text,"camera_id":item.camera_id,
        "observed_at":item.timestamp.isoformat(),"latitude":item.latitude,"longitude":item.longitude,
        "plate_confidence":item.plate_confidence,"source_track_id":item.source_track_id,
        "direction":item.direction,"status":item.status.value,"source_type":item.source}

@router.get("/sightings/recent")
def recent(request: Request, limit: int = Query(50, ge=1, le=500), camera_id: str | None = None):
    repository = request.app.state.storage.sightings
    rows = repository.get_recent(limit, camera_id)
    return [_serialize(x) for x in rows]

@router.get("/sightings/{plate_text}")
def by_plate(plate_text: str, request: Request, start_time: datetime | None = None, end_time: datetime | None = None):
    rows = request.app.state.storage.sightings.get_by_plate(plate_text)
    rows = [x for x in rows if (start_time is None or x.timestamp >= start_time) and (end_time is None or x.timestamp <= end_time)]
    return [_serialize(x) for x in rows]

@router.get("/cameras")
def cameras(request: Request):
    return [{"camera_id":x.camera_id,"name":x.name,"latitude":x.latitude,"longitude":x.longitude,
        "road_name":x.road_name,"sector":x.sector,"direction":x.direction,"is_active":x.is_active}
        for x in request.app.state.storage.cameras.list()]

@router.get("/trajectory/{plate_text}")
def trajectory(plate_text: str, request: Request, start_time: datetime | None = None, end_time: datetime | None = None):
    storage = request.app.state.storage
    camera_map = {x.camera_id:x for x in storage.cameras.list()}
    reconstructor = TrajectoryReconstructor(storage.sightings, camera_map, CameraTopology(DEMO_EDGES))
    return reconstructor.reconstruct(plate_text, start_time, end_time).to_dict()
