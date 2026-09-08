from backend.alerts.models import AlertStatus
from backend.alerts.repository import InMemoryAlertRepository
from backend.alerts.service import AlertService
from cv.alerts.restricted_zone import (
    RestrictedZoneConfig,
    RestrictedZoneEngine,
    RestrictedZoneObservation,
    RestrictedZonePolicy,
    RestrictedZoneState,
)
from cv.detection.models import BoundingBox


def test_runner_annotation_marks_inside_track_red():
    import numpy as np

    from scripts.run_restricted_zone_demo import annotate_frame

    class Track:
        track_id = 9
        bbox = BoundingBox(20, 20, 40, 60)

    class Tracker:
        @staticmethod
        def display_class(_track_id): return "Car"

    zone = RestrictedZonePolicy("Z", "Restricted", ((0, 0), (1, 0), (1, 1), (0, 1)))
    config = RestrictedZoneConfig("CAM", (zone,))
    zone_engine = RestrictedZoneEngine(config)
    zone_engine.update((RestrictedZoneObservation(9, Track.bbox, "Car", .9),),
        source_time_s=0, width=100, height=100)
    image = annotate_frame(np.zeros((100, 100, 3), dtype=np.uint8), (Track(),),
        Tracker(), zone_engine, config, hide_zones=True)
    assert image[20, 20, 2] > image[20, 20, 1]

ZONE=RestrictedZonePolicy("RZ-1","SECURE ENTRY",((.2,.2),(.8,.2),(.8,.8),(.2,.8)),
    entry_confirm_seconds=2.,grace_seconds=1.,allowed_vehicle_classes=("Ambulance",),allowed_plates=("KA02MN1826",))


def engine(): return RestrictedZoneEngine(RestrictedZoneConfig("CAM-1",(ZONE,)))
def observation(x=.5,vehicle_class="Car",plate=None,status=None,track=7):
    return RestrictedZoneObservation(track,BoundingBox((x-.05)*100,30,(x+.05)*100,60),vehicle_class,.9,plate,.95,status)
def step(item,time,*rows): return item.update(rows,source_time_s=time,width=100,height=100)


def test_outside_and_short_contact_do_not_alert():
    item=engine(); assert step(item,0,observation(x=.95))==()
    assert step(item,1,observation())==() and step(item,2.9,observation())==()


def test_unauthorized_persistent_entry_alerts_and_deduplicates():
    item=engine(); repository=InMemoryAlertRepository(); service=AlertService(repository)
    step(item,0,observation()); created=service.process_restricted_zone_events(step(item,2,observation()))
    updated=service.process_restricted_zone_events(step(item,3,observation()))
    assert len(created)==len(updated)==1 and created[0].alert_id==updated[0].alert_id and len(repository.list())==1


def test_allowed_class_or_accepted_allowed_plate_authorizes():
    for row in (observation(vehicle_class="Ambulance"),observation(plate="KA02MN1826",status="ACCEPTED")):
        item=engine(); assert step(item,0,row)==() and step(item,3,row)==()
        assert item.state(7,"RZ-1") is RestrictedZoneState.OUTSIDE


def test_unconfirmed_plate_cannot_authorize():
    item=engine(); row=observation(plate="KA02MN1826",status="PROVISIONAL")
    step(item,0,row); event=step(item,2,row)[0]
    assert event.plate_text is None and event.authorization_reason=="PLATE_NOT_ACCEPTED"


def test_leave_clears_and_reentry_creates_new_lifecycle():
    item=engine(); repository=InMemoryAlertRepository(); service=AlertService(repository)
    step(item,0,observation()); first=service.process_restricted_zone_events(step(item,2,observation()))[0]
    cleared=service.process_restricted_zone_events(step(item,3,observation(x=.95)))[0]
    assert cleared.status is AlertStatus.CLEARED
    step(item,4,observation()); second=service.process_restricted_zone_events(step(item,6,observation()))[0]
    assert second.alert_id!=first.alert_id and len(repository.list())==2


def test_alert_payload_keeps_optional_plate_absent_safely():
    item=engine(); repository=InMemoryAlertRepository(); service=AlertService(repository)
    step(item,0,observation()); payload=service.process_restricted_zone_events(step(item,2,observation()))[0].to_dict()
    assert payload["plate_text"] is None and payload["plate_confidence"] is None
    assert payload["zone_label"]=="SECURE ENTRY" and payload["authorization_status"]=="UNAUTHORIZED"
