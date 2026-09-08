from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from fastapi.testclient import TestClient

from backend.alerts.blacklisted_vehicle import BlacklistedVehicleAlertHandler
from backend.alerts.models import AlertType
from backend.alerts.repository import InMemoryAlertRepository
from backend.alerts.watchlist import WatchlistEntry, WatchlistSeverity
from backend.alerts.watchlist_repository import InMemoryWatchlistRepository
from backend.api.app import create_app
from cv.ocr.models import FusionStatus

NOW=datetime(2026,9,8,10,tzinfo=UTC)


def entry(plate="KA02MN1826",active=True,expires_at=None):
    return WatchlistEntry("WL-1",plate,"Demo security watch",WatchlistSeverity.CRITICAL,
        "DEMO WATCHLIST ENTRY",active,NOW-timedelta(days=1),expires_at,"Synthetic demo record")


def completed(status=FusionStatus.ACCEPTED,plate="KA02MN1826",track=19):
    return SimpleNamespace(camera_id="CAM-01",track_id=track,timestamp=NOW,
        fused_plate_result=SimpleNamespace(status=status,plate_text=plate,confidence=.9566))


def setup(item=None):
    watchlist=InMemoryWatchlistRepository(); alerts=InMemoryAlertRepository()
    if item: watchlist.save(item)
    return BlacklistedVehicleAlertHandler(alerts,watchlist),alerts


def test_accepted_exact_match_creates_evidence_rich_alert():
    handler,alerts=setup(entry()); created=handler.handle_completed(completed())
    assert len(created)==1 and len(alerts.list())==1
    payload=created[0].to_dict()
    assert payload["alert_type"]==AlertType.BLACKLISTED_VEHICLE.value
    assert payload["plate_text"]=="KA02MN1826" and payload["plate_confidence"]==.9566
    assert payload["watchlist_id"]=="WL-1" and payload["demo_watchlist_entry"] is True


def test_no_match_or_one_character_mismatch_creates_no_alert():
    handler,_=setup(entry())
    assert handler.handle_completed(completed(plate="KA02MN1825"))==()
    assert handler.handle_completed(completed(plate="DL01AA0001"))==()


def test_unconfirmed_and_low_confidence_results_never_match():
    handler,_=setup(entry())
    for status in (FusionStatus.UNKNOWN,FusionStatus.LOW_CONFIDENCE):
        assert handler.handle_completed(completed(status=status))==()


def test_inactive_or_expired_entry_never_matches():
    for item in (entry(active=False),entry(expires_at=NOW-timedelta(seconds=1))):
        handler,_=setup(item); assert handler.handle_completed(completed())==()


def test_duplicate_same_watchlist_camera_track_creates_one_alert():
    handler,alerts=setup(entry()); handler.handle_completed(completed()); handler.handle_completed(completed())
    assert len(alerts.list())==1


def test_watchlist_api_crud_uses_exact_plate_text():
    class Stream:
        def stop(self): pass
    app=create_app(stream_service=Stream())
    with TestClient(app) as client:
        response=client.post("/api/watchlist",json={"plate_text":"KA02MN1826","reason":"Demo",
            "severity":"HIGH","source":"DEMO WATCHLIST ENTRY"})
        assert response.status_code==201 and response.json()["plate_text"]=="KA02MN1826"
        identifier=response.json()["watchlist_id"]
        assert client.get("/api/watchlist").json()[0]["demo_entry"] is True
        assert client.patch(f"/api/watchlist/{identifier}",json={"notes":"Presentation only"}).status_code==200
        assert client.delete(f"/api/watchlist/{identifier}").json()["active"] is False
