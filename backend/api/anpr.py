import json

import cv2
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from backend.streaming.factory import DEFAULT_EVENTS, DEFAULT_OUTPUT

router = APIRouter(prefix="/api/anpr", tags=["anpr"])


class ProcessingRateRequest(BaseModel):
    rate: str


def _service(request: Request):
    service = request.app.state.anpr_stream
    service.start()
    return service


@router.get("/status")
def status(request: Request, mode: str = "LIVE_ANALYSIS"):
    if mode == "RECORDED_ANALYSIS":
        capture = cv2.VideoCapture(str(DEFAULT_OUTPUT))
        fps = capture.get(cv2.CAP_PROP_FPS) if capture.isOpened() else 0.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if capture.isOpened() else 0
        capture.release()
        payload = {
            "backend_online": True, "running": False, "frame": 0, "fps": 0.0,
            "source": DEFAULT_OUTPUT.name, "camera_id": "CAM-DEMO-01",
            "pipeline_status": "PREPROCESSED", "error": None,
            "playback_state": "PAUSED", "current_frame": 0,
            "total_frames": total, "current_time_s": 0.0,
            "duration_s": total / fps if fps else 0.0,
            "source_fps": fps, "playback_fps": fps,
            "inference_fps": 0.0, "latest_analyzed_frame": total - 1,
            "mode": "RECORDED_ANALYSIS",
            "processing_rate": "1.25x",
        }
        payload["database"] = request.app.state.storage.status()
        return payload
    if mode != "LIVE_ANALYSIS":
        raise HTTPException(400, "mode must be LIVE_ANALYSIS or RECORDED_ANALYSIS")
    payload = _service(request).status().to_dict()
    payload["database"] = request.app.state.storage.status()
    return payload


@router.get("/events")
def events(request: Request):
    return [event.to_dict() for event in _service(request).recent_events()]


@router.get("/recognition")
def recognition(request: Request):
    return _service(request).current_recognition()


@router.get("/stream")
def stream(request: Request):
    return StreamingResponse(
        _service(request).mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )


@router.get("/recorded-video")
def recorded_video():
    if not DEFAULT_OUTPUT.is_file():
        raise HTTPException(404, "Recorded analysis video has not been generated")
    return FileResponse(DEFAULT_OUTPUT, media_type="video/mp4")


@router.get("/recorded-events")
def recorded_events():
    if not DEFAULT_EVENTS.is_file():
        raise HTTPException(404, "Recorded analysis event sidecar has not been generated")
    return json.loads(DEFAULT_EVENTS.read_text(encoding="utf-8"))


@router.post("/play")
def play(request: Request):
    service = _service(request)
    service.resume()
    return service.status().to_dict()


@router.post("/pause")
def pause(request: Request):
    service = _service(request)
    service.pause()
    return service.status().to_dict()


@router.post("/restart")
def restart(request: Request):
    service = _service(request)
    service.restart()
    return service.status().to_dict()


@router.post("/processing-rate")
def processing_rate(payload: ProcessingRateRequest, request: Request):
    service = _service(request)
    try:
        service.set_processing_rate(payload.rate)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return service.status().to_dict()
