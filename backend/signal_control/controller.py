from datetime import UTC, datetime
from math import floor

from backend.signal_control.models import (
    ApproachSchedulingState,
    ApproachTrafficState,
    SignalControlConfig,
    SignalCyclePlan,
    SignalDecision,
)
from backend.signal_control.pressure import pressure_score, reason_codes


class AdaptiveSignalController:
    """Explainable plan generator; it does not actuate physical signal hardware."""

    def __init__(self, config: SignalControlConfig | None = None) -> None:
        self.config = config or SignalControlConfig()

    def plan(self, intersection_id: str, approaches: tuple[ApproachTrafficState, ...],
             scheduling: dict[str, ApproachSchedulingState] | None = None,
             generated_at: datetime | None = None) -> SignalCyclePlan:
        if not intersection_id.strip(): raise ValueError("intersection_id must not be empty")
        if any(x.intersection_id != intersection_id for x in approaches):
            raise ValueError("all approaches must belong to the requested intersection")
        if len({x.approach_id for x in approaches}) != len(approaches):
            raise ValueError("approach_id values must be unique")
        scheduling = scheduling or {}; decisions = []
        for approach in approaches:
            pressure = pressure_score(approach, self.config)
            state = scheduling.get(approach.approach_id, ApproachSchedulingState())
            fairness = (state.consecutive_cycles_skipped >= self.config.max_skip_cycles
                        or state.time_since_last_green_s >= self.config.max_red_wait_s)
            effective = min(1.0, pressure + (self.config.fairness_boost if fairness else 0.0))
            span = self.config.max_green_s - self.config.min_green_s
            green = floor(self.config.min_green_s + pressure * span + .5)
            green = max(self.config.min_green_s, min(self.config.max_green_s, green))
            decisions.append(SignalDecision(intersection_id, approach.approach_id, pressure,
                effective, green, fairness, reason_codes(approach, pressure, fairness, self.config)))
        ordered = tuple(sorted(decisions, key=lambda x: (-x.effective_priority, x.approach_id)))
        overhead = len(ordered) * (self.config.yellow_s + self.config.all_red_s)
        return SignalCyclePlan(intersection_id, generated_at or datetime.now(UTC), ordered,
                               sum(x.green_duration_s for x in ordered) + overhead)
