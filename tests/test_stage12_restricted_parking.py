from datetime import UTC, datetime

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.alerts.models import AlertStatus, AlertType
from backend.alerts.repository import InMemoryAlertRepository
from backend.alerts.service import AlertService
from backend.api.app import create_app
from cv.alerts.parking_state import (
    ParkingObservation,
    RestrictedParkingConfig,
    RestrictedParkingZone,
)
from cv.alerts.restricted_parking import ParkingState, RestrictedParkingEngine
from cv.common.frame import FramePacket
from cv.detection.models import BoundingBox, VehicleDetection
from cv.traffic_state.tracker import TrafficByteTracker

ZONE=RestrictedParkingZone("Z","NO PARKING",((.1,.1),(.9,.1),(.9,.9),(.1,.9)))


def config(**kwargs): return RestrictedParkingConfig("CAM",(ZONE,),**kwargs)
def observation(track=1,x=.5,y=.5,confidence=.9,vehicle_class="Car",raw="Sedan"):
    return ParkingObservation(track,BoundingBox((x-.05)*100,(y-.1)*100,(x+.05)*100,y*100),confidence,vehicle_class,raw,.9)
def step(engine,time,*items): return engine.update(items,source_time_s=time,width=100,height=100)


def test_drive_through_zone_creates_no_violation():
    engine=RestrictedParkingEngine(config())
    assert not step(engine,0,observation(x=.2)); assert not step(engine,1,observation(x=.5)); assert not step(engine,2,observation(x=.8))
    assert engine.state(1,"Z") is ParkingState.ON_RESTRICTED_SURFACE


def test_one_second_stop_does_not_alert():
    engine=RestrictedParkingEngine(config())
    step(engine,0,observation()); assert not step(engine,1,observation())
    assert engine.state(1,"Z") is ParkingState.ON_RESTRICTED_SURFACE


def test_stationary_beyond_threshold_creates_one_lifecycle_not_per_frame():
    engine=RestrictedParkingEngine(config()); repository=InMemoryAlertRepository(); service=AlertService(repository)
    step(engine,0,observation()); step(engine,1,observation()); assert not service.process_parking_events(step(engine,4.9,observation()))
    created=service.process_parking_events(step(engine,5,observation())); updated=service.process_parking_events(step(engine,6,observation()))
    assert len(created)==len(updated)==1 and created[0].alert_id==updated[0].alert_id
    assert len(repository.list())==1 and repository.list()[0].status is AlertStatus.UPDATED


def test_footpath_mode_uses_continuous_occupancy_without_stationary_motion():
    engine=RestrictedParkingEngine(config(footpath_enforcement_mode=True,violation_min_seconds=1.5))
    assert not step(engine,0,observation(x=.45)); assert not step(engine,1,observation(x=.55))
    events=step(engine,1.5,observation(x=.65))
    assert events[0].state is ParkingState.ILLEGALLY_PARKED
    assert "FOOTPATH_OCCUPANCY_CONFIRMED" in events[0].decision_reason_codes


def test_default_mode_still_requires_stationary_motion():
    engine=RestrictedParkingEngine(config(violation_min_seconds=1.5))
    step(engine,0,observation(x=.45)); step(engine,1,observation(x=.55))
    assert not step(engine,2,observation(x=.65))
    assert engine.state(1,"Z") is ParkingState.ON_RESTRICTED_SURFACE


def test_short_occlusion_preserves_same_alert():
    engine=RestrictedParkingEngine(config(track_grace_seconds=1.5)); repository=InMemoryAlertRepository(); service=AlertService(repository)
    step(engine,0,observation()); step(engine,1,observation()); first=service.process_parking_events(step(engine,5,observation()))[0]
    assert not service.process_parking_events(step(engine,6)); resumed=service.process_parking_events(step(engine,6.4,observation()))[0]
    assert resumed.alert_id==first.alert_id and resumed.status is AlertStatus.UPDATED


def test_long_occlusion_clears_alert():
    engine=RestrictedParkingEngine(config(track_grace_seconds=1.5)); repository=InMemoryAlertRepository(); service=AlertService(repository)
    step(engine,0,observation()); step(engine,1,observation()); service.process_parking_events(step(engine,5,observation()))
    cleared=service.process_parking_events(step(engine,7))[0]
    assert cleared.status is AlertStatus.CLEARED and cleared.cleared_at==7


def test_moving_or_leaving_clears_active_alert():
    engine=RestrictedParkingEngine(config()); repository=InMemoryAlertRepository(); service=AlertService(repository)
    step(engine,0,observation()); step(engine,1,observation()); service.process_parking_events(step(engine,5,observation()))
    cleared=service.process_parking_events(step(engine,6,observation(x=.95)))[0]
    assert cleared.status is AlertStatus.CLEARED


def test_vehicle_outside_polygon_never_alerts():
    engine=RestrictedParkingEngine(config())
    for time in (0,1,5,10): assert not step(engine,time,observation(x=.95))


def test_two_vehicles_produce_independent_alerts():
    engine=RestrictedParkingEngine(config()); repository=InMemoryAlertRepository(); service=AlertService(repository)
    step(engine,0,observation(1),observation(2,x=.6)); step(engine,1,observation(1),observation(2,x=.6))
    alerts=service.process_parking_events(step(engine,5,observation(1),observation(2,x=.6)))
    assert len(alerts)==2 and len({x.track_id for x in alerts})==2


def test_duplicate_zone_ids_are_invalid():
    with pytest.raises(ValueError): RestrictedParkingConfig("CAM",(ZONE,ZONE))


def test_source_time_not_processing_rate_controls_decision():
    def run(times):
        engine=RestrictedParkingEngine(config())
        events=[]
        for time in times: events.extend(step(engine,time,observation()))
        return bool([x for x in events if x.state is ParkingState.ILLEGALLY_PARKED])
    assert run((0,1,2,3,4,5))==run((0,1,5))==True


def test_tracker_class_consensus_is_used_for_parking_observation():
    tracker=TrafficByteTracker(); image=np.zeros((100,100,3),np.uint8)
    def detect(index,raw,confidence):
        frame=FramePacket("CAM",index,datetime(2026,1,1,tzinfo=UTC),image)
        return VehicleDetection(frame,BoundingBox(20,20,40,60),raw,confidence,"fake","1")
    track=tracker.update([detect(0,"Three-wheeler",.9)])[0]
    tracker.update([detect(1,"Three-wheeler",.9)]); tracker.update([detect(2,"Hatchback",.4)])
    assert tracker.display_class(track.track_id)=="Auto-rickshaw"


def test_alert_api_lists_filters_and_gets_by_id():
    class Stream:
        def stop(self): pass
    app=create_app(stream_service=Stream()); repository=app.state.alert_repository
    engine=RestrictedParkingEngine(config()); service=AlertService(repository)
    step(engine,0,observation()); step(engine,1,observation()); alert=service.process_parking_events(step(engine,5,observation()))[0]
    client=TestClient(app)
    assert client.get("/api/alerts").json()[0]["alert_type"]==AlertType.RESTRICTED_PARKING.value
    assert client.get("/api/alerts/active").status_code==200
    assert client.get(f"/api/alerts/{alert.alert_id}").json()["track_id"]==1


def test_pending_then_illegal_policy_state_and_api_evidence_fields():
    engine=RestrictedParkingEngine(config()); repository=InMemoryAlertRepository(); service=AlertService(repository)
    step(engine,0,observation()); step(engine,1,observation()); assert not step(engine,2.5,observation())
    assert engine.state(1,"Z") is ParkingState.STATIONARY_PENDING
    alert=service.process_parking_events(step(engine,5,observation()))[0]
    payload=alert.to_dict()
    assert payload["parking_state"]=="ILLEGALLY_PARKED"
    assert payload["plate_text"] is None
    for field in ("vehicle_class_confidence","detection_confidence","motion_score","stationary_duration_s",
            "ground_contact_fraction","restricted_overlap","surface_class","surface_confidence","policy_mode",
            "decision_confidence","reason_codes","first_seen_at","confirmed_at","rule_version"):
        assert field in payload


def test_cleared_alert_keeps_same_lifecycle_key_and_id():
    engine=RestrictedParkingEngine(config()); repository=InMemoryAlertRepository(); service=AlertService(repository)
    step(engine,0,observation()); step(engine,1,observation()); created=service.process_parking_events(step(engine,5,observation()))[0]
    cleared=service.process_parking_events(step(engine,6,observation(x=.95)))[0]
    assert cleared.alert_id==created.alert_id and cleared.lifecycle_key==created.lifecycle_key
    assert cleared.to_dict()["parking_state"]=="CLEARED"
