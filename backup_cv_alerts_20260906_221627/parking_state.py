from dataclasses import dataclass
from math import isfinite

from cv.detection.models import BoundingBox

Point=tuple[float,float]


@dataclass(frozen=True,slots=True)
class RestrictedParkingZone:
    zone_id:str; label:str; polygon:tuple[Point,...]; enabled:bool=True
    def __post_init__(self):
        if not self.zone_id.strip(): raise ValueError("zone_id is required")
        if len(self.polygon)<3: raise ValueError("restricted parking polygon requires at least three points")
        if any(len(point)!=2 or any(not isfinite(value) or not 0<=value<=1 for value in point) for point in self.polygon):
            raise ValueError("zone coordinates must be finite and within 0..1")


@dataclass(frozen=True,slots=True)
class RestrictedParkingConfig:
    camera_id:str; zones:tuple[RestrictedParkingZone,...]; stationary_motion_threshold:float=.006
    stationary_min_seconds:float=2.; violation_min_seconds:float=5.; track_grace_seconds:float=1.5
    minimum_detection_confidence:float=.35; surface_mode:str="POLICY_DEFINED"
    restricted_surface_classes:tuple[str,...]=("SIDEWALK",); minimum_surface_confidence:float=.55
    semantic_refresh_interval_s:float=5.; stabilization_frames:int=3
    ground_contact_fraction:float=.25; minimum_restricted_overlap:float=.40
    semantic_enabled:bool=False; semantic_model_path:str|None=None; semantic_input_size:tuple[int,int]=(512,1024)
    semantic_provider:str="torchscript_cityscapes"
    def __post_init__(self):
        if not self.camera_id.strip(): raise ValueError("camera_id is required")
        identifiers=[x.zone_id for x in self.zones]
        if len(identifiers)!=len(set(identifiers)): raise ValueError("zone IDs must be unique per camera")
        values=(self.stationary_motion_threshold,self.stationary_min_seconds,self.violation_min_seconds,self.track_grace_seconds)
        if any(not isfinite(x) or x<=0 for x in values): raise ValueError("parking thresholds must be finite and positive")
        if not 0<=self.minimum_detection_confidence<=1: raise ValueError("minimum_detection_confidence must be within 0..1")
        if self.surface_mode not in {"SEMANTIC","POLICY_DEFINED","HYBRID"}: raise ValueError("invalid restricted surface mode")
        if not self.restricted_surface_classes: raise ValueError("at least one restricted surface class is required")
        if any(x.upper() not in {"ROAD","SIDEWALK","OTHER"} for x in self.restricted_surface_classes): raise ValueError("invalid restricted surface class")
        if not 0<=self.minimum_surface_confidence<=1 or not 0<self.ground_contact_fraction<=1 or not 0<=self.minimum_restricted_overlap<=1:
            raise ValueError("surface confidence and geometry fractions must be within 0..1")
        if self.semantic_refresh_interval_s<=0 or self.stabilization_frames<1: raise ValueError("invalid semantic stabilization settings")
        if self.surface_mode in {"SEMANTIC","HYBRID"} and not self.semantic_enabled: raise ValueError("semantic must be enabled for SEMANTIC/HYBRID mode")
        if self.semantic_enabled and not self.semantic_model_path: raise ValueError("semantic model_path is required when semantic is enabled")
        if len(self.semantic_input_size)!=2 or any(not isinstance(x,int) or x<=0 for x in self.semantic_input_size): raise ValueError("semantic input_size must contain two positive integers")
        if self.semantic_provider not in {"torchscript_cityscapes","huggingface_segformer"}: raise ValueError("invalid semantic provider")


@dataclass(frozen=True,slots=True)
class ParkingObservation:
    track_id:int; bbox:BoundingBox; detection_confidence:float; vehicle_class:str
    raw_vehicle_class:str|None=None; class_confidence:float|None=None
