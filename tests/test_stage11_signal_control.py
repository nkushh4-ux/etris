from dataclasses import FrozenInstanceError
from datetime import UTC, datetime
from math import inf, nan

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.signal_control import router
from backend.signal_control.controller import AdaptiveSignalController
from backend.signal_control.demo import (
    build_demo_plan,
)
from backend.signal_control.models import (
    ApproachSchedulingState,
    ApproachTrafficState,
    NormalizationReferences,
    PressureWeights,
    SignalControlConfig,
)
from backend.signal_control.pressure import pressure_score


def state(approach="A", vehicle=0, queue=0, occupancy=0, wait=0, arrival=0, heavy=0):
    return ApproachTrafficState("I", approach, vehicle, queue, occupancy, wait,
                                arrival, heavy, datetime(2026,9,5,tzinfo=UTC))


def plan(*approaches, scheduling=None, config=None):
    return AdaptiveSignalController(config).plan("I", tuple(approaches), scheduling,
                                                  datetime(2026,9,5,tzinfo=UTC))


def test_pressure_is_bounded_zero_to_one():
    config = SignalControlConfig()
    assert pressure_score(state(), config) == 0
    assert pressure_score(state(vehicle=500,queue=500,occupancy=1,wait=500,arrival=500,heavy=500), config) == 1


def test_zero_traffic_gets_minimum_green():
    assert plan(state()).ordered_phases[0].green_duration_s == 10


def test_saturated_traffic_gets_maximum_green():
    decision = plan(state(vehicle=50,queue=25,occupancy=1,wait=60,arrival=30,heavy=10)).ordered_phases[0]
    assert decision.green_duration_s == 60


@pytest.mark.parametrize(("field","value"), [("queue",20),("wait",50),("occupancy",.9)])
def test_demand_components_increase_pressure(field, value):
    values = {field:value}
    assert pressure_score(state(**values), SignalControlConfig()) > pressure_score(state(), SignalControlConfig())


def test_heavy_vehicles_contribute_without_dominating():
    config = SignalControlConfig()
    heavy = pressure_score(state(heavy=10), config)
    queue = pressure_score(state(queue=25), config)
    assert 0 < heavy < queue


def test_green_time_stays_inside_custom_bounds():
    config = SignalControlConfig(min_green_s=15,max_green_s=35)
    assert plan(state(),config=config).ordered_phases[0].green_duration_s == 15
    saturated = state(vehicle=999,queue=999,occupancy=1,wait=999,arrival=999,heavy=999)
    assert plan(saturated,config=config).ordered_phases[0].green_duration_s == 35


def test_ties_are_ordered_by_approach_id():
    result = plan(state("WEST"),state("EAST"),state("NORTH"))
    assert [x.approach_id for x in result.ordered_phases] == ["EAST","NORTH","WEST"]


@pytest.mark.parametrize("scheduling", [
    ApproachSchedulingState(consecutive_cycles_skipped=2),
    ApproachSchedulingState(time_since_last_green_s=120),
])
def test_fairness_boost_at_skip_or_wait_threshold(scheduling):
    decision = plan(state(),scheduling={"A":scheduling}).ordered_phases[0]
    assert decision.fairness_applied
    assert decision.effective_priority == .25
    assert "FAIRNESS_STARVATION_GUARD" in decision.reason_codes


def test_fairness_priority_is_bounded_and_green_is_safe():
    busy = state(vehicle=50,queue=25,occupancy=1,wait=60,arrival=30,heavy=10)
    decision = plan(busy,scheduling={"A":ApproachSchedulingState(9,999)}).ordered_phases[0]
    assert decision.effective_priority == 1
    assert decision.green_duration_s == 60


def test_every_approach_is_scheduled_so_none_starves():
    result = plan(state("A",vehicle=50),state("B"),state("C"),
        scheduling={"C":ApproachSchedulingState(2,120)})
    assert {x.approach_id for x in result.ordered_phases} == {"A","B","C"}
    assert all(x.green_duration_s >= 10 for x in result.ordered_phases)


def test_cycle_duration_includes_each_transition_overhead():
    result = plan(state("A"),state("B"))
    assert result.cycle_duration_s == 20 + 2 * (3 + 1)


def test_first_demo_is_led_by_north():
    result = build_demo_plan()
    assert result.ordered_phases[0].approach_id == "NORTH"


def test_changed_demo_raises_east_priority_and_green():
    before, after = build_demo_plan(), build_demo_plan(True)
    old = next(x for x in before.ordered_phases if x.approach_id=="EAST")
    new = next(x for x in after.ordered_phases if x.approach_id=="EAST")
    assert new.effective_priority > old.effective_priority
    assert new.green_duration_s > old.green_duration_s
    assert after.ordered_phases[0].approach_id == "EAST"


@pytest.mark.parametrize("kwargs", [{"vehicle":-1},{"queue":-1},{"wait":nan},{"arrival":inf},{"heavy":-1}])
def test_invalid_numeric_input_is_rejected(kwargs):
    with pytest.raises(ValueError): state(**kwargs)


@pytest.mark.parametrize("occupancy", [-.01,1.01,nan,inf])
def test_invalid_occupancy_is_rejected(occupancy):
    with pytest.raises(ValueError): state(occupancy=occupancy)


def test_empty_approach_list_is_safe():
    result = plan()
    assert result.ordered_phases == () and result.cycle_duration_s == 0


def test_models_and_configuration_are_immutable_and_validated():
    with pytest.raises(FrozenInstanceError): state().vehicle_count = 4
    with pytest.raises(ValueError): PressureWeights(vehicle_count=.5)
    with pytest.raises(ValueError): NormalizationReferences(queue_length=0)


def test_reason_codes_are_specific():
    decision = plan(state(queue=25,occupancy=.8,wait=60,arrival=30,heavy=5)).ordered_phases[0]
    assert {"HIGH_QUEUE","HIGH_WAIT","HIGH_OCCUPANCY","HIGH_ARRIVAL_RATE","HEAVY_VEHICLE_LOAD"} <= set(decision.reason_codes)


def test_api_response_schema_and_synthetic_labels():
    app = FastAPI(); app.include_router(router); client = TestClient(app)
    demo = client.get("/api/signal-control/demo")
    changed = client.get("/api/signal-control/demo/change")
    assert demo.status_code == 200 and demo.json()["data_source"] == "DEMO_SYNTHETIC"
    assert changed.json()["scenario"] == "EAST_DEMAND_SURGE"
    body = {"intersection_id":"I","approaches":[{"approach_id":"NORTH",
        "vehicle_count":10,"queue_length":3,"lane_occupancy":.2,
        "average_waiting_time_s":12,"arrival_rate_vpm":5,"heavy_vehicle_count":0,
        "timestamp":"2026-09-05T10:00:00Z"}]}
    response = client.post("/api/signal-control/plan",json=body)
    payload = response.json()
    assert response.status_code == 200
    assert payload["intersection_id"] == "I" and payload["recommendation_only"] is True
    assert len(payload["ordered_phases"]) == 1


def test_mismatched_intersection_and_duplicate_approaches_rejected():
    controller = AdaptiveSignalController()
    with pytest.raises(ValueError): controller.plan("OTHER",(state(),))
    with pytest.raises(ValueError): plan(state(),state())
