"""Deterministic synthetic Stage 11 signal-control scenarios."""

from datetime import UTC, datetime

from backend.signal_control.controller import AdaptiveSignalController
from backend.signal_control.models import ApproachTrafficState, SignalCyclePlan

DEMO_TIMESTAMP = datetime(2026, 9, 7, 9, 0, tzinfo=UTC)
DEMO_INTERSECTION_ID = "DEMO-INT-01"


def _state(approach: str, vehicle: float, queue: float, occupancy: float,
           wait: float, arrival: float, heavy: float) -> ApproachTrafficState:
    return ApproachTrafficState(DEMO_INTERSECTION_ID, approach, vehicle, queue,
        occupancy, wait, arrival, heavy, DEMO_TIMESTAMP)


DEMO_APPROACHES = (
    _state("NORTH", 48, 20, .82, 48, 24, 4),
    _state("SOUTH", 10, 3, .22, 12, 7, 0),
    _state("EAST", 22, 8, .45, 24, 12, 1),
    _state("WEST", 31, 12, .60, 32, 15, 2),
)

CHANGED_DEMO_APPROACHES = tuple(
    _state("EAST", 50, 24, .95, 55, 28, 6) if item.approach_id == "EAST" else item
    for item in DEMO_APPROACHES
)


def build_demo_plan(changed: bool = False) -> SignalCyclePlan:
    approaches = CHANGED_DEMO_APPROACHES if changed else DEMO_APPROACHES
    return AdaptiveSignalController().plan(DEMO_INTERSECTION_ID, approaches, generated_at=DEMO_TIMESTAMP)
