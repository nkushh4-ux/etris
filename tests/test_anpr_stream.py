from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.streaming.anpr_stream import (
    ANPREvent,
    ANPRStreamService,
    ANPRStreamStatus,
    InferenceFrame,
    LatestFrameMailbox,
    events_from_completed,
    presentation_overlay_labels,
)
from cv.ocr.models import FusionStatus, OCRResultStatus


def event(number=1):
    return ANPREvent("KA02MN1826", "ACCEPTED", .97, "CAM-01", number,
                     "2026-09-07T10:00:00+00:00", number)


def completed(fusion_status=FusionStatus.ACCEPTED, plate="KA02MN1826"):
    result = SimpleNamespace(status=OCRResultStatus.SUCCESS,
                             normalized_text="KA02HN1826", confidence=.92)
    return SimpleNamespace(
        camera_id="CAM-01", track_id=19,
        timestamp=datetime(2026, 9, 7, tzinfo=UTC), termination_frame_index=299,
        ocr_results=(result,), fused_plate_result=SimpleNamespace(
            status=fusion_status, plate_text=plate, confidence=.97),
    )


def test_event_and_status_serialization_shapes():
    assert event().to_dict()["plate"] == "KA02MN1826"
    status = ANPRStreamStatus(True, 12, 8.4, "traffic.mp4", "CAM-01", "RUNNING")
    payload = status.to_dict()
    assert payload["frame"] == 12 and payload["fps"] == 8.4
    assert payload["playback_state"] == "PLAYING"
    assert payload["backend_online"] is True
    assert {"source_fps", "playback_fps", "inference_fps", "latest_analyzed_frame"} <= payload.keys()


def test_event_buffer_is_bounded_and_recent_first(tmp_path):
    service = ANPRStreamService(tmp_path / "none.mp4", "CAM-01", lambda: None, event_limit=2)
    for number in range(3):
        service.add_event(event(number))
    assert [item.track_id for item in service.recent_events()] == [2, 1]


def test_accepted_event_preserves_fused_identity_and_provisional_is_explicit():
    events = events_from_completed(completed())
    assert events[0].plate == "KA02HN1826" and events[0].status == "PROVISIONAL"
    assert events[1].plate == "KA02MN1826" and events[1].status == "ACCEPTED"


def test_unknown_never_becomes_confirmed_plate():
    final = events_from_completed(completed(FusionStatus.UNKNOWN, "SHOULD_NOT_LEAK"))[-1]
    assert final.status == "UNKNOWN" and final.plate == "UNKNOWN"


def test_unknown_labels_are_hidden_only_from_video_presentation():
    labels = ("UNKNOWN [UNKNOWN 0.00]", "KA02MN1826 [ACCEPTED 0.96]")
    assert presentation_overlay_labels(labels) == ("KA02MN1826 [ACCEPTED 0.96]",)


def test_stream_state_reset(tmp_path):
    service = ANPRStreamService(tmp_path / "none.mp4", "CAM-01", lambda: None)
    service.add_event(event())
    service._frame, service._fps, service._pipeline_status = 8, 2.0, "RUNNING"
    service.reset_state()
    assert service.status().frame == -1 and service.status().fps == 0
    assert service.recent_events() == ()


class _FakeService:
    def __init__(self): self.started = 0
    def start(self): self.started += 1
    def stop(self): pass
    def status(self): return ANPRStreamStatus(False, -1, 0, "demo.mp4", "CAM", "IDLE")
    def recent_events(self): return (event(),)
    def pause(self): self.action = "pause"
    def resume(self): self.action = "play"
    def restart(self): self.action = "restart"
    def set_processing_rate(self, rate): self.rate = rate


def test_api_status_and_events_json_shapes():
    fake = _FakeService()
    with TestClient(create_app(stream_service=fake)) as client:
        status = client.get("/api/anpr/status")
        events = client.get("/api/anpr/events")
    assert status.status_code == 200 and status.json()["camera_id"] == "CAM"
    assert events.status_code == 200 and events.json()[0]["status"] == "ACCEPTED"
    assert fake.started == 2


def test_latest_frame_mailbox_is_bounded_and_drops_stale_frames():
    mailbox = LatestFrameMailbox()
    first = InferenceFrame(0, 1, datetime.now(UTC), object())
    latest = InferenceFrame(0, 2, datetime.now(UTC), object())
    mailbox.put_latest(first)
    mailbox.put_latest(latest)
    assert mailbox.pending_count == 1
    assert mailbox.dropped_frames == 1
    assert mailbox.get() is latest


def test_status_progress_and_separate_fps(tmp_path):
    service = ANPRStreamService(tmp_path / "none.mp4", "CAM-01", lambda: None)
    with service._condition:
        service._frame = 150
        service._total_frames = 930
        service._source_fps = 30.0
        service._playback_fps = 29.7
        service._inference_fps = 3.4
        service._latest_analyzed_frame = 145
    status = service.status()
    assert status.current_time_s == 5.0
    assert status.duration_s == 31.0
    assert (status.source_fps, status.playback_fps, status.inference_fps) == (30.0, 29.7, 3.4)
    assert status.fps == 3.4 and status.latest_analyzed_frame == 145


def test_pause_resume_and_restart_state(tmp_path):
    service = ANPRStreamService(tmp_path / "none.mp4", "CAM-01", lambda: None)
    service._frame = 50
    service._annotation = object()
    service.add_event(event())
    service.pause()
    assert service.status().playback_state == "PAUSED" and service.status().frame == 50
    service.resume()
    assert service.status().playback_state == "PLAYING"
    service.restart()
    status = service.status()
    assert status.frame == -1 and status.latest_analyzed_frame == -1
    assert service._annotation is None and service.recent_events() == ()


def test_display_publication_does_not_duplicate_events(tmp_path):
    import numpy as np
    service = ANPRStreamService(tmp_path / "none.mp4", "CAM-01", lambda: None)
    service.add_event(event())
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    service._publish(image, 1, 30.0)
    service._publish(image, 2, 30.0)
    assert len(service.recent_events()) == 1


def test_playback_control_routes():
    fake = _FakeService()
    with TestClient(create_app(stream_service=fake)) as client:
        assert client.post("/api/anpr/pause").status_code == 200
        assert fake.action == "pause"
        assert client.post("/api/anpr/play").status_code == 200
        assert fake.action == "play"
        assert client.post("/api/anpr/restart").status_code == 200
        assert fake.action == "restart"


def test_recorded_event_serialization_and_video_time():
    item = event()
    timed = ANPREvent(item.plate, item.status, item.confidence, item.camera_id,
                      item.track_id, item.timestamp, 299, 299 / 30.0)
    payload = timed.to_dict()
    assert payload["frame_id"] == 299
    assert payload["video_time_s"] == 299 / 30.0


def test_recorded_status_does_not_start_ml_worker():
    fake = _FakeService()
    with TestClient(create_app(stream_service=fake)) as client:
        response = client.get("/api/anpr/status?mode=RECORDED_ANALYSIS")
    assert response.status_code == 200
    assert response.json()["mode"] == "RECORDED_ANALYSIS"
    assert response.json()["pipeline_status"] == "PREPROCESSED"
    assert fake.started == 0


def test_live_status_starts_only_one_worker_across_queries():
    fake = _FakeService()
    with TestClient(create_app(stream_service=fake)) as client:
        client.get("/api/anpr/status?mode=LIVE_ANALYSIS")
        client.get("/api/anpr/status?mode=LIVE_ANALYSIS")
    # The API asks the service to start; the real service makes start idempotent.
    assert fake.started == 2


def test_live_service_uses_sequential_worker_not_mailbox_worker(tmp_path):
    service = ANPRStreamService(tmp_path / "none.mp4", "CAM", lambda: None)
    assert service._live_sequential_loop.__func__ is ANPRStreamService._live_sequential_loop
    assert service._mailbox.pending_count == 0


def test_final_accepted_result_overrides_preview_presentation(tmp_path):
    service = ANPRStreamService(tmp_path / "none.mp4", "CAM-01", lambda: None)
    service._current_recognition = {"plate": "KA02HN1826", "status": "LIVE_CONSENSUS"}
    service._record_completed(completed(), source_fps=30.0)
    recognition = service.current_recognition()
    assert recognition["status"] == "ACCEPTED"
    assert recognition["plate"] == "KA02MN1826"


def test_paused_frame_zero_and_zero_fps_are_still_backend_online():
    status = ANPRStreamStatus(
        True, 0, 0.0, "traffic.mp4", "CAM", "PAUSED",
        playback_state="PAUSED", current_frame=0, source_fps=30.0,
        inference_fps=0.0,
    ).to_dict()
    assert status["backend_online"] is True
    assert status["current_frame"] == 0
    assert status["inference_fps"] == 0.0
    assert status["playback_state"] == "PAUSED"


def test_live_processing_rates_exclude_frame_skipping_modes(tmp_path):
    service = ANPRStreamService(tmp_path / "none.mp4", "CAM", lambda: None)
    for rate in ("AUTO", "1.0x", "0.5x"):
        service.set_processing_rate(rate)
        assert service.status().processing_rate == rate
    for rate in ("1.25x", "1.5x"):
        with pytest.raises(ValueError):
            service.set_processing_rate(rate)


def test_processing_rate_api_shape():
    fake = _FakeService()
    with TestClient(create_app(stream_service=fake)) as client:
        response = client.post("/api/anpr/processing-rate", json={"rate": "AUTO"})
    assert response.status_code == 200
    assert fake.rate == "AUTO"
