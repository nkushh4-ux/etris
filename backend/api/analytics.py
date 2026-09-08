from datetime import datetime

from fastapi import APIRouter, HTTPException, Request

from backend.analytics.service import TrafficAnalyticsService
from backend.trajectory.demo import DEMO_EDGES

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


def _service(request: Request) -> TrafficAnalyticsService:
    storage = request.app.state.storage
    return TrafficAnalyticsService(storage.sightings,
        {x.camera_id: x for x in storage.cameras.list()}, DEMO_EDGES)


def _window(start_time: datetime | None, end_time: datetime | None):
    default_start, default_end = TrafficAnalyticsService.recent_window()
    start, end = start_time or default_start, end_time or default_end
    if end < start: raise HTTPException(422, "end_time must not precede start_time")
    return start, end


def _rows(method, start_time, end_time):
    start, end = _window(start_time, end_time)
    return [x.to_dict() for x in method(start, end)]


@router.get("/summary")
def summary(request: Request, start_time: datetime | None = None, end_time: datetime | None = None):
    start, end = _window(start_time, end_time)
    return _service(request).summary(start, end).to_dict()


@router.get("/camera-volume")
def camera_volume(request: Request, start_time: datetime | None = None, end_time: datetime | None = None):
    service = _service(request); return _rows(service.camera_volume, start_time, end_time)


@router.get("/od")
def od(request: Request, start_time: datetime | None = None, end_time: datetime | None = None):
    service = _service(request); return _rows(service.od_flows, start_time, end_time)


@router.get("/segments")
def segments(request: Request, start_time: datetime | None = None, end_time: datetime | None = None):
    service = _service(request); return _rows(service.segment_analytics, start_time, end_time)


@router.get("/congestion")
def congestion(request: Request, start_time: datetime | None = None, end_time: datetime | None = None):
    start, end = _window(start_time, end_time)
    snapshots = _service(request).congestion(start, end)
    ingestor = getattr(request.app.state, "congestion_ingestor", None)
    if ingestor is not None:
        ingestor.process_once((start.isoformat(), end.isoformat()), snapshots, now=end.timestamp())
    return [snapshot.to_dict() for snapshot in snapshots]


@router.get("/heatmap")
def heatmap(request: Request, start_time: datetime | None = None, end_time: datetime | None = None):
    service = _service(request); return _rows(service.heatmap, start_time, end_time)


@router.get("/routes")
def routes(request: Request, start_time: datetime | None = None, end_time: datetime | None = None):
    service = _service(request); return _rows(service.routes, start_time, end_time)
