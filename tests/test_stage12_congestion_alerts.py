from backend.alerts.models import AlertStatus, AlertType
from backend.alerts.repository import InMemoryAlertRepository
from backend.alerts.service import AlertService
from backend.analytics.models import CongestionLevel, CongestionResult


def snapshot(level,ratio=None,samples=3):
    values={CongestionLevel.FREE_FLOW:1.1,CongestionLevel.MODERATE:1.4,
        CongestionLevel.CONGESTED:1.8,CongestionLevel.SEVERE:2.4}
    return CongestionResult("CAM-01","CAM-03",samples,100.,None if ratio is None and samples<3 else 100.*values.get(level,1),
        ratio if ratio is not None else values.get(level),level)


def setup():
    repository=InMemoryAlertRepository(); return AlertService(repository),repository


def send(service,level,count,start=1,samples=3):
    emitted=[]
    for offset in range(count): emitted.extend(service.process_congestion_snapshots((snapshot(level,samples=samples),),now=start+offset))
    return emitted


def test_free_flow_and_moderate_do_not_alert():
    service,repository=setup(); send(service,CongestionLevel.FREE_FLOW,4); send(service,CongestionLevel.MODERATE,4,start=5)
    assert repository.list()==()


def test_one_congested_snapshot_does_not_alert():
    service,repository=setup(); send(service,CongestionLevel.CONGESTED,1)
    assert repository.list()==()


def test_persistent_congested_creates_one_deduplicated_alert():
    service,repository=setup(); send(service,CongestionLevel.CONGESTED,3); send(service,CongestionLevel.CONGESTED,3,start=4)
    assert len(repository.list())==1 and repository.list()[0].alert_type is AlertType.CONGESTION
    assert repository.list()[0].metadata["sample_count"]==3


def test_persistent_severe_creates_one_severe_delay():
    service,repository=setup(); send(service,CongestionLevel.SEVERE,2)
    assert len(repository.list())==1 and repository.list()[0].alert_type is AlertType.SEVERE_DELAY


def test_congestion_escalates_only_after_two_consecutive_severe_snapshots():
    service,repository=setup(); send(service,CongestionLevel.CONGESTED,3)
    first=send(service,CongestionLevel.SEVERE,1,start=4)
    assert first==[] and repository.list()[0].status is not AlertStatus.CLEARED
    emitted=send(service,CongestionLevel.SEVERE,1,start=5)
    assert [item.status for item in emitted]==[AlertStatus.CLEARED,AlertStatus.ACTIVE]
    active=[item for item in repository.list() if item.status is not AlertStatus.CLEARED]
    assert len(active)==1 and active[0].alert_type is AlertType.SEVERE_DELAY


def test_three_recovery_snapshots_clear_active_alert_without_flapping():
    service,_repository=setup(); send(service,CongestionLevel.CONGESTED,3)
    assert send(service,CongestionLevel.MODERATE,2,start=4)==[]
    cleared=send(service,CongestionLevel.FREE_FLOW,1,start=6)
    assert len(cleared)==1 and cleared[0].status is AlertStatus.CLEARED
    assert cleared[0].metadata["cleared_at"]==6


def test_insufficient_data_neither_creates_nor_clears():
    service,repository=setup(); send(service,CongestionLevel.INSUFFICIENT_DATA,5,samples=2)
    assert repository.list()==()
    send(service,CongestionLevel.SEVERE,2,start=6); active=repository.list()[0]
    assert send(service,CongestionLevel.INSUFFICIENT_DATA,5,start=8,samples=2)==[]
    assert repository.get(active.alert_id).status is not AlertStatus.CLEARED


def test_alert_serialization_uses_only_stage10_evidence():
    service,_=setup(); payload=send(service,CongestionLevel.CONGESTED,3)[0].to_dict()
    assert payload["segment_id"]=="CAM-01->CAM-03" and payload["delay_ratio"]==1.8
    assert payload["observed_travel_time_s"]==180. and "route_id" not in payload
