import json
from pathlib import Path

from cv.alerts.parking_state import RestrictedParkingConfig, RestrictedParkingZone


def load_restricted_parking_config(path:str|Path)->RestrictedParkingConfig:
    data=json.loads(Path(path).read_text(encoding="utf-8")); settings=data.get("settings",{}); surface=data.get("restricted_surface",{})
    semantic=surface.get("semantic",{}); geometry=surface.get("geometry",{})
    zones=tuple(RestrictedParkingZone(x["zone_id"],x.get("label",x["zone_id"]),tuple(map(tuple,x["polygon"])),x.get("enabled",True))
                for x in data.get("restricted_parking_zones",()))
    return RestrictedParkingConfig(data["camera_id"],zones,settings.get("stationary_motion_threshold",.006),
        settings.get("stationary_min_seconds",2.),settings.get("violation_min_seconds",5.),
        settings.get("track_grace_seconds",1.5),settings.get("minimum_detection_confidence",.35),surface.get("mode","POLICY_DEFINED"),
        tuple(semantic.get("restricted_classes",("SIDEWALK",))),semantic.get("minimum_surface_confidence",.55),
        semantic.get("refresh_interval_s",5.),semantic.get("stabilization_frames",3),geometry.get("ground_contact_fraction",.25),
        geometry.get("minimum_restricted_overlap",.40),semantic.get("enabled",False),semantic.get("model_path"),
        tuple(semantic.get("input_size",(512,1024))),semantic.get("provider","torchscript_cityscapes"))
