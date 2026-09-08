from datetime import UTC, datetime, timedelta

import pytest

from backend.analytics.congestion import CongestionThresholds, classify_congestion
from backend.analytics.demo import (
    DEMO_WINDOW_END,
    DEMO_WINDOW_START,
    build_demo_analytics,
)
from backend.analytics.models import CongestionLevel
from backend.analytics.service import TrafficAnalyticsService
from backend.domain.cameras import CameraNode
from backend.domain.sightings import SightingStatus, VehicleSighting
from backend.trajectory.demo import DEMO_CAMERAS
from backend.trajectory.models import CameraEdge
from backend.trajectory.repository import InMemorySightingRepository

START = datetime(2026, 9, 5, 10, tzinfo=UTC)
END = START + timedelta(hours=1)
CAMERAS = {x.camera_id: x for x in (
    CameraNode("A", "Alpha", 12.0, 77.0), CameraNode("B", "Beta", 12.1, 77.1),
    CameraNode("C", "Gamma", 12.2, 77.2), CameraNode("D", "Delta", 12.3, 77.3),
)}
EDGES = (CameraEdge("A", "B", 1000, 100, 500), CameraEdge("B", "C", 2000, 100, 500))


def row(number, plate, camera, seconds, status=SightingStatus.CONFIRMED):
    node = CAMERAS[camera]
    return VehicleSighting(str(number), plate, camera, START + timedelta(seconds=seconds),
        node.latitude, node.longitude, .9, number, status=status, source="TEST")


def service(*rows, baselines=None):
    repository = InMemorySightingRepository()
    for item in rows: repository.add(item)
    return TrafficAnalyticsService(repository, CAMERAS, EDGES,
        baseline_travel_seconds=baselines)


def test_camera_volume_and_unique_vehicles_confirmed_only():
    result = service(row(1,"P1","A",0), row(2,"P1","A",10), row(3,"P2","A",20),
        row(4,"NOISE","A",30,SightingStatus.LOW_CONFIDENCE)).camera_volume(START, END)
    alpha = result[0]
    assert (alpha.vehicle_count, alpha.unique_vehicle_count, alpha.vehicles_per_hour) == (3, 2, 3)


def test_od_aggregation_and_same_camera_exclusion():
    result = service(row(1,"P1","A",0),row(2,"P1","B",120),row(3,"P2","A",0),
        row(4,"P2","B",130),row(5,"P3","A",0),row(6,"P3","A",20)).od_flows(START,END)
    assert result == (result[0],) and result[0].to_dict() == {
        "origin_camera":"A","destination_camera":"B","vehicle_count":2}


def test_segment_speed_median_and_invalid_transitions_excluded():
    analytics = service(row(1,"P1","A",0),row(2,"P1","B",100),
        row(3,"P2","A",0),row(4,"P2","B",200),
        row(5,"FAST","A",0),row(6,"FAST","B",10),
        row(7,"UNKNOWN","A",0),row(8,"UNKNOWN","C",200))
    segment = analytics.segment_analytics(START,END)[0]
    assert segment.vehicle_count == 2
    assert segment.median_travel_time_s == 150
    assert segment.mean_speed_kmh == pytest.approx(27)
    assert segment.median_speed_kmh == pytest.approx(27)


@pytest.mark.parametrize(("ratio","expected"), [
    (1.1,CongestionLevel.FREE_FLOW),(1.3,CongestionLevel.MODERATE),
    (1.8,CongestionLevel.CONGESTED),(2.2,CongestionLevel.SEVERE)])
def test_congestion_classes(ratio, expected):
    assert classify_congestion(3, ratio, CongestionThresholds()) is expected


def test_congestion_insufficient_data():
    assert classify_congestion(2, 3, CongestionThresholds()) is CongestionLevel.INSUFFICIENT_DATA


def test_congestion_uses_configured_baseline():
    analytics = service(*(item for n in range(3) for item in
        (row(n*2+1,f"P{n}","A",0),row(n*2+2,f"P{n}","B",200))), baselines={("A","B"):160})
    result = analytics.congestion(START,END)[0]
    assert result.travel_time_ratio == 1.25
    assert result.classification is CongestionLevel.MODERATE


def test_heatmap_normalization_and_zero_dataset():
    values = service(row(1,"P1","A",0),row(2,"P2","A",1),row(3,"P3","B",2)).heatmap(START,END)
    assert [x.normalized_intensity for x in values] == [1, .5, 0, 0]
    assert all(x.normalized_intensity == 0 for x in service().heatmap(START,END))


def test_route_density_normalization_includes_empty_edges():
    values = service(row(1,"P1","A",0),row(2,"P1","B",100)).routes(START,END)
    assert [(x.source,x.target,x.vehicle_count,x.normalized_density) for x in values] == [
        ("A","B",1,1),("B","C",0,0)]


def test_network_summary_and_empty_behavior():
    analytics = service(row(1,"P1","A",0),row(2,"P1","B",100))
    result = analytics.summary(START,END)
    assert (result.total_sightings,result.unique_vehicles,result.active_cameras,
            result.inter_camera_movements) == (2,1,2,1)
    assert result.busiest_camera == "B"  # deterministic tie break
    assert result.busiest_segment == "A->B"
    empty = service().summary(START,END)
    assert empty.total_sightings == 0 and empty.busiest_camera is None
    assert empty.average_valid_speed_kmh is None


def test_deterministic_ordering():
    analytics = service(row(1,"P2","B",130),row(2,"P2","A",0),
        row(3,"P1","B",120),row(4,"P1","A",0))
    assert analytics.od_flows(START,END) == analytics.od_flows(START,END)
    assert [(x.origin_camera,x.destination_camera) for x in analytics.od_flows(START,END)] == [("A","B")]


def test_repository_contract_supplies_chronological_confirmed_rows():
    repository = InMemorySightingRepository()
    repository.add(row(2,"P","A",20)); repository.add(row(1,"P","A",10))
    repository.add(row(3,"LOW","A",5,SightingStatus.LOW_CONFIDENCE))
    assert [x.sighting_id for x in repository.get_between(START,END)] == ["1","2"]


def test_window_validation():
    with pytest.raises(ValueError): service().camera_volume(END, START)


def test_models_are_json_ready_and_immutable():
    volume = service(row(1,"P","A",0)).camera_volume(START,END)[0]
    assert volume.to_dict()["window_start"].endswith("+00:00")
    with pytest.raises(Exception): volume.vehicle_count = 9


def test_demo_has_thirty_vehicles_busy_route_and_is_deterministic():
    first = build_demo_analytics(); second = build_demo_analytics()
    summary = first.summary(DEMO_WINDOW_START, DEMO_WINDOW_END)
    assert summary.unique_vehicles == 30
    assert summary.busiest_camera == "CAM-03"
    assert summary.busiest_segment == "CAM-01->CAM-03"
    congestion = {(x.source_camera, x.target_camera): x for x in first.congestion(DEMO_WINDOW_START, DEMO_WINDOW_END)}
    assert congestion[("CAM-01","CAM-03")].classification is CongestionLevel.SEVERE
    assert summary.to_dict() == second.summary(DEMO_WINDOW_START, DEMO_WINDOW_END).to_dict()


def test_all_analytics_api_routes_return_json_shapes():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from backend.api.analytics import router

    demo = build_demo_analytics()
    class Cameras:
        def list(self): return tuple(DEMO_CAMERAS.values())
    class Storage: pass
    storage = Storage(); storage.sightings = demo.repository; storage.cameras = Cameras()
    app = FastAPI(); app.state.storage = storage; app.include_router(router)
    query = "?start_time=2026-09-07T08:00:00Z&end_time=2026-09-07T10:00:00Z"
    client = TestClient(app)
    for path in ("summary","camera-volume","od","segments","congestion","heatmap","routes"):
        response = client.get(f"/api/analytics/{path}{query}")
        assert response.status_code == 200
        assert isinstance(response.json(), dict if path == "summary" else list)
