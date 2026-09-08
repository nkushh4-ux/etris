import json
from pathlib import Path

from cv.traffic_state.models import ApproachRegion, TrafficStateConfig


def load_traffic_state_config(path: str | Path) -> TrafficStateConfig:
    data=json.loads(Path(path).read_text(encoding="utf-8"))
    approaches=tuple(ApproachRegion(approach_id=x["approach_id"],label=x["label"],
        traffic_polygon=tuple(map(tuple,x["traffic_polygon"])),queue_polygon=tuple(map(tuple,x["queue_polygon"])),
        entry_line=tuple(map(tuple,x["entry_line"])),exit_line=tuple(map(tuple,x["exit_line"])) if x.get("exit_line") else None,
        entry_direction=tuple(x.get("entry_direction",(0,1)))) for x in data["approaches"])
    settings=data.get("settings",{})
    return TrafficStateConfig(data["intersection_id"],approaches,
        settings.get("arrival_window_s",30),settings.get("stationary_motion_threshold",.012),
        settings.get("queue_stationary_min_s",1.25),settings.get("controller_update_interval_s",2),
        settings.get("track_expiry_s",2))
