from dataclasses import dataclass
from enum import Enum
from math import isfinite

import cv2
import numpy as np

from cv.detection.models import BoundingBox


class RestrictedZoneMode(str,Enum): DENY_BY_DEFAULT="DENY_BY_DEFAULT"
class RestrictedZoneState(str,Enum): OUTSIDE="OUTSIDE"; ENTRY_PENDING="ENTRY_PENDING"; VIOLATION="VIOLATION"; CLEARED="CLEARED"


@dataclass(frozen=True,slots=True)
class RestrictedZonePolicy:
    zone_id:str; label:str; polygon:tuple[tuple[float,float],...]; enabled:bool=True
    mode:RestrictedZoneMode=RestrictedZoneMode.DENY_BY_DEFAULT; entry_confirm_seconds:float=1.; grace_seconds:float=1.5
    allowed_vehicle_classes:tuple[str,...]=(); allowed_plates:tuple[str,...]=()
    def __post_init__(self):
        if not self.zone_id.strip() or not self.label.strip() or len(self.polygon)<3: raise ValueError("valid zone ID, label, and polygon are required")
        if any(len(point)!=2 or any(not isfinite(value) or not 0<=value<=1 for value in point) for point in self.polygon): raise ValueError("zone coordinates must be within 0..1")
        if self.entry_confirm_seconds<=0 or self.grace_seconds<0: raise ValueError("entry confirmation must be positive and grace non-negative")


@dataclass(frozen=True,slots=True)
class RestrictedZoneConfig:
    camera_id:str; zones:tuple[RestrictedZonePolicy,...]
    def __post_init__(self):
        if not self.camera_id.strip(): raise ValueError("camera_id is required")
        if len({zone.zone_id for zone in self.zones})!=len(self.zones): raise ValueError("zone IDs must be unique")


@dataclass(frozen=True,slots=True)
class RestrictedZoneObservation:
    track_id:int; bbox:BoundingBox; vehicle_class:str; vehicle_class_confidence:float|None=None
    plate_text:str|None=None; plate_confidence:float|None=None; plate_status:str|None=None


@dataclass(frozen=True,slots=True)
class RestrictedZoneEvent:
    camera_id:str; zone_id:str; zone_label:str; track_id:int; state:RestrictedZoneState
    vehicle_class:str; vehicle_class_confidence:float|None; plate_text:str|None; plate_confidence:float|None
    entry_time:float; confirmed_time:float|None; cleared_time:float|None; inside_duration_s:float
    authorization_status:str; authorization_reason:str; rule_version:str; reason_codes:tuple[str,...]
    decision_confidence:str


@dataclass(slots=True)
class _ZoneMemory:
    entry_time:float; last_seen:float; confirmed_time:float|None; observation:RestrictedZoneObservation


class RestrictedZoneEngine:
    rule_version="RESTRICTED-ZONE-1.0"
    def __init__(self,config:RestrictedZoneConfig): self.config=config; self._memory={}
    def update(self,observations,*,source_time_s:float,width:int,height:int):
        events=[]; seen=set()
        for observation in sorted(observations,key=lambda item:item.track_id):
            for zone in self.config.zones:
                if not zone.enabled: continue
                key=(observation.track_id,zone.zone_id); memory=self._memory.get(key)
                inside=_inside(zone,observation.bbox,width,height)
                if not inside:
                    if memory and memory.confirmed_time is not None: events.append(self._event(zone,memory,RestrictedZoneState.CLEARED,source_time_s))
                    self._memory.pop(key,None); continue
                seen.add(key); authorized,reason=_authorization(zone,observation)
                if authorized:
                    if memory and memory.confirmed_time is not None: events.append(self._event(zone,memory,RestrictedZoneState.CLEARED,source_time_s,"AUTHORIZED",reason))
                    self._memory.pop(key,None); continue
                if memory is None:
                    self._memory[key]=_ZoneMemory(source_time_s,source_time_s,None,observation); continue
                memory.last_seen=source_time_s; memory.observation=observation
                if memory.confirmed_time is None and source_time_s-memory.entry_time>=zone.entry_confirm_seconds: memory.confirmed_time=source_time_s
                if memory.confirmed_time is not None: events.append(self._event(zone,memory,RestrictedZoneState.VIOLATION,source_time_s,"UNAUTHORIZED",reason))
        for key,memory in list(self._memory.items()):
            zone=next((item for item in self.config.zones if item.zone_id==key[1]),None)
            if key not in seen and zone and source_time_s-memory.last_seen>zone.grace_seconds:
                if memory.confirmed_time is not None: events.append(self._event(zone,memory,RestrictedZoneState.CLEARED,source_time_s))
                del self._memory[key]
        return tuple(events)
    def state(self,track_id,zone_id,source_time_s=None):
        memory=self._memory.get((track_id,zone_id))
        if memory is None: return RestrictedZoneState.OUTSIDE
        return RestrictedZoneState.VIOLATION if memory.confirmed_time is not None else RestrictedZoneState.ENTRY_PENDING
    def _event(self,zone,memory,state,now,status="UNAUTHORIZED",reason="NOT_IN_ALLOWLIST"):
        observation=memory.observation; accepted=observation.plate_status=="ACCEPTED" and bool(observation.plate_text)
        reasons=("RESTRICTED_ZONE_ENTRY_CONFIRMED",reason) if state is RestrictedZoneState.VIOLATION else ("RESTRICTED_ZONE_EXITED",)
        return RestrictedZoneEvent(self.config.camera_id,zone.zone_id,zone.label,observation.track_id,state,
            observation.vehicle_class,observation.vehicle_class_confidence,observation.plate_text if accepted else None,
            observation.plate_confidence if accepted else None,memory.entry_time,memory.confirmed_time,
            now if state is RestrictedZoneState.CLEARED else None,max(0.,now-memory.entry_time),status,reason,
            self.rule_version,reasons,"HIGH" if observation.vehicle_class_confidence is not None else "MEDIUM")


def _inside(zone,bbox,width,height):
    point=((bbox.x1+bbox.x2)/2/max(width,1),bbox.y2/max(height,1))
    polygon=np.asarray(zone.polygon,np.float32)
    return cv2.pointPolygonTest(polygon,point,False)>=0


def _authorization(zone,observation):
    if observation.vehicle_class in zone.allowed_vehicle_classes: return True,"ALLOWED_VEHICLE_CLASS"
    if observation.plate_status=="ACCEPTED" and observation.plate_text:
        return (True,"ACCEPTED_PLATE_ALLOWLIST") if observation.plate_text in zone.allowed_plates else (False,"ACCEPTED_PLATE_NOT_ALLOWED")
    if observation.plate_text: return False,"PLATE_NOT_ACCEPTED"
    return False,"NO_ACCEPTED_PLATE"
