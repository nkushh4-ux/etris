from backend.signal_control.models import ApproachTrafficState, SignalControlConfig


def pressure_score(state: ApproachTrafficState, config: SignalControlConfig) -> float:
    refs, weights = config.references, config.weights
    components = (
        (weights.vehicle_count, state.vehicle_count / refs.vehicle_count),
        (weights.queue_length, state.queue_length / refs.queue_length),
        (weights.lane_occupancy, state.lane_occupancy),
        (weights.average_waiting_time, state.average_waiting_time_s / refs.average_waiting_time_s),
        (weights.arrival_rate, state.arrival_rate_vpm / refs.arrival_rate_vpm),
        (weights.heavy_vehicle_count, state.heavy_vehicle_count / refs.heavy_vehicle_count),
    )
    return max(0.0, min(1.0, sum(weight * min(value, 1.0) for weight, value in components)))


def reason_codes(state: ApproachTrafficState, pressure: float,
                 fairness_applied: bool, config: SignalControlConfig) -> tuple[str, ...]:
    refs = config.references; reasons = []
    if state.queue_length / refs.queue_length >= .7: reasons.append("HIGH_QUEUE")
    if state.average_waiting_time_s / refs.average_waiting_time_s >= .7: reasons.append("HIGH_WAIT")
    if state.lane_occupancy >= .7: reasons.append("HIGH_OCCUPANCY")
    if state.arrival_rate_vpm / refs.arrival_rate_vpm >= .7: reasons.append("HIGH_ARRIVAL_RATE")
    if state.heavy_vehicle_count / refs.heavy_vehicle_count >= .5: reasons.append("HEAVY_VEHICLE_LOAD")
    if fairness_applied: reasons.append("FAIRNESS_STARVATION_GUARD")
    if pressure < .25: reasons.append("LOW_DEMAND")
    return tuple(reasons or ("BALANCED_DEMAND",))
