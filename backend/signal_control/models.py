from dataclasses import asdict, dataclass, field
from datetime import datetime
from math import isfinite
from typing import Any


def _finite_non_negative(name: str, value: float) -> None:
    if not isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class PressureWeights:
    vehicle_count: float = .25
    queue_length: float = .25
    lane_occupancy: float = .15
    average_waiting_time: float = .20
    arrival_rate: float = .10
    heavy_vehicle_count: float = .05

    def __post_init__(self) -> None:
        values = tuple(asdict(self).values())
        if any(not isfinite(x) or x < 0 for x in values):
            raise ValueError("pressure weights must be finite and non-negative")
        if abs(sum(values) - 1.0) > 1e-9:
            raise ValueError("pressure weights must sum to 1.0")


@dataclass(frozen=True, slots=True)
class NormalizationReferences:
    vehicle_count: float = 50
    queue_length: float = 25
    average_waiting_time_s: float = 60
    arrival_rate_vpm: float = 30
    heavy_vehicle_count: float = 10

    def __post_init__(self) -> None:
        if any(not isfinite(x) or x <= 0 for x in asdict(self).values()):
            raise ValueError("normalization references must be finite and positive")


@dataclass(frozen=True, slots=True)
class SignalControlConfig:
    weights: PressureWeights = field(default_factory=PressureWeights)
    references: NormalizationReferences = field(default_factory=NormalizationReferences)
    min_green_s: int = 10
    max_green_s: int = 60
    yellow_s: int = 3
    all_red_s: int = 1
    max_skip_cycles: int = 2
    max_red_wait_s: float = 120
    fairness_boost: float = .25

    def __post_init__(self) -> None:
        if self.min_green_s < 0 or self.max_green_s < self.min_green_s:
            raise ValueError("green timing bounds are invalid")
        if self.yellow_s < 0 or self.all_red_s < 0:
            raise ValueError("transition times must be non-negative")
        if self.max_skip_cycles < 1 or self.max_red_wait_s <= 0:
            raise ValueError("fairness thresholds must be positive")
        if not isfinite(self.fairness_boost) or not 0 <= self.fairness_boost <= 1:
            raise ValueError("fairness_boost must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class ApproachTrafficState:
    intersection_id: str
    approach_id: str
    vehicle_count: float
    queue_length: float
    lane_occupancy: float
    average_waiting_time_s: float
    arrival_rate_vpm: float
    heavy_vehicle_count: float
    timestamp: datetime

    def __post_init__(self) -> None:
        if not self.intersection_id.strip() or not self.approach_id.strip():
            raise ValueError("intersection_id and approach_id must not be empty")
        for name in ("vehicle_count", "queue_length", "average_waiting_time_s",
                     "arrival_rate_vpm", "heavy_vehicle_count"):
            _finite_non_negative(name, float(getattr(self, name)))
        if not isfinite(self.lane_occupancy) or not 0 <= self.lane_occupancy <= 1:
            raise ValueError("lane_occupancy must be finite and between 0 and 1")
        if not isinstance(self.timestamp, datetime):
            raise ValueError("timestamp must be a datetime")


@dataclass(frozen=True, slots=True)
class ApproachSchedulingState:
    consecutive_cycles_skipped: int = 0
    time_since_last_green_s: float = 0

    def __post_init__(self) -> None:
        if self.consecutive_cycles_skipped < 0:
            raise ValueError("consecutive_cycles_skipped must be non-negative")
        _finite_non_negative("time_since_last_green_s", self.time_since_last_green_s)


@dataclass(frozen=True, slots=True)
class SignalDecision:
    intersection_id: str
    approach_id: str
    pressure: float
    effective_priority: float
    green_duration_s: int
    fairness_applied: bool
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self); data["reason_codes"] = list(self.reason_codes); return data


@dataclass(frozen=True, slots=True)
class SignalCyclePlan:
    intersection_id: str
    generated_at: datetime
    ordered_phases: tuple[SignalDecision, ...]
    cycle_duration_s: int
    recommendation_only: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"intersection_id": self.intersection_id, "generated_at": self.generated_at.isoformat(),
                "ordered_phases": [x.to_dict() for x in self.ordered_phases],
                "cycle_duration_s": self.cycle_duration_s,
                "recommendation_only": self.recommendation_only}
