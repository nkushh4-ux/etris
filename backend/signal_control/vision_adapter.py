from datetime import datetime

from backend.signal_control.controller import AdaptiveSignalController
from backend.signal_control.models import ApproachTrafficState, SignalCyclePlan
from cv.traffic_state.models import TrafficStateFrame

REAL_VIDEO_ESTIMATE = "REAL_VIDEO_ESTIMATE"


def approach_states_from_video(frame: TrafficStateFrame, intersection_id:str,
                               timestamp:datetime) -> tuple[ApproachTrafficState,...]:
    return tuple(ApproachTrafficState(intersection_id,x.approach_id,x.active_vehicle_count,
        x.queue_length,x.image_space_occupancy,x.average_waiting_time_s,x.arrival_rate_vpm,
        x.heavy_vehicle_count,timestamp) for x in frame.approaches)


def plan_from_video(frame:TrafficStateFrame, intersection_id:str, timestamp:datetime,
                    controller:AdaptiveSignalController) -> SignalCyclePlan:
    return controller.plan(intersection_id,approach_states_from_video(frame,intersection_id,timestamp),generated_at=timestamp)
