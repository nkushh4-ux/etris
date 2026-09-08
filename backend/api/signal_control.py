from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.signal_control.controller import AdaptiveSignalController
from backend.signal_control.demo import build_demo_plan
from backend.signal_control.models import ApproachSchedulingState, ApproachTrafficState

router = APIRouter(prefix="/api/signal-control", tags=["signal-control"])
PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRAFFIC_DEMO_DIR = PROJECT_ROOT / "runs" / "signal_demo_full"


class ApproachPayload(BaseModel):
    approach_id: str
    vehicle_count: float
    queue_length: float
    lane_occupancy: float
    average_waiting_time_s: float
    arrival_rate_vpm: float
    heavy_vehicle_count: float
    timestamp: datetime
    consecutive_cycles_skipped: int = 0
    time_since_last_green_s: float = 0


class PlanPayload(BaseModel):
    intersection_id: str
    approaches: list[ApproachPayload] = Field(default_factory=list)


def _plan(payload: PlanPayload):
    try:
        approaches = tuple(ApproachTrafficState(payload.intersection_id, item.approach_id,
            item.vehicle_count, item.queue_length, item.lane_occupancy,
            item.average_waiting_time_s, item.arrival_rate_vpm,
            item.heavy_vehicle_count, item.timestamp) for item in payload.approaches)
        scheduling = {item.approach_id: ApproachSchedulingState(
            item.consecutive_cycles_skipped, item.time_since_last_green_s) for item in payload.approaches}
        return AdaptiveSignalController().plan(payload.intersection_id, approaches, scheduling).to_dict()
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/demo")
def demo():
    return {"data_source":"DEMO_SYNTHETIC", "scenario":"BASELINE",
            "plan":build_demo_plan().to_dict()}


@router.get("/demo/change")
def demo_change():
    return {"data_source":"DEMO_SYNTHETIC", "scenario":"EAST_DEMAND_SURGE",
            "plan":build_demo_plan(True).to_dict()}


@router.post("/plan")
def plan(payload: PlanPayload):
    return _plan(payload)


@router.get("/traffic-demo/events")
def traffic_demo_events():
    path = TRAFFIC_DEMO_DIR / "traffic_state_events.json"
    if not path.is_file(): raise HTTPException(404, "Preprocessed traffic-state events are unavailable")
    return FileResponse(path, media_type="application/json")


@router.get("/traffic-demo/video")
def traffic_demo_video():
    path = TRAFFIC_DEMO_DIR / "traffic_signal_annotated.mp4"
    if not path.is_file(): raise HTTPException(404, "Preprocessed traffic-state video is unavailable")
    return FileResponse(path, media_type="video/mp4")
