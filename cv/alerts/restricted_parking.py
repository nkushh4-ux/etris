from dataclasses import dataclass
from enum import Enum
from math import hypot

from cv.alerts.parking_state import ParkingObservation, RestrictedParkingConfig
from cv.alerts.restricted_surface import (
    RestrictedSurfaceEvidence,
    RestrictedSurfaceMode,
    RestrictedSurfaceResolver,
)


class ParkingState(str,Enum):
    OUTSIDE="OUTSIDE"; ON_RESTRICTED_SURFACE="ON_RESTRICTED_SURFACE"
    STATIONARY_PENDING="STATIONARY_PENDING"; ILLEGALLY_PARKED="ILLEGALLY_PARKED"; CLEARED="CLEARED"; UNCERTAIN="UNCERTAIN"


@dataclass(frozen=True,slots=True)
class ParkingEvent:
    camera_id:str; zone_id:str; track_id:int; state:ParkingState; source_time_s:float
    vehicle_class:str; raw_vehicle_class:str|None; class_confidence:float|None; detection_confidence:float
    entered_zone_at:float|None; stationary_since:float|None; violation_started_at:float|None; stationary_seconds:float
    surface_class:str="POLICY_DEFINED"; raw_surface_class:str|None=None; surface_confidence:float=1.
    restricted_overlap:float=0.; ground_contact_fraction:float=.25; motion_score:float=0.
    policy_mode:str="POLICY_DEFINED"; rule_version:str="RPZ-2.0"; decision_confidence:str="LOW"
    decision_reason_codes:tuple[str,...]=()

    def to_dict(self): return {name:(value.value if isinstance(value,Enum) else value) for name,value in ((field,getattr(self,field)) for field in self.__dataclass_fields__)}


@dataclass(slots=True)
class _Memory:
    state:ParkingState; anchor:tuple[float,float]; last_seen_s:float; entered_s:float
    stationary_since:float|None; violation_s:float|None; observation:ParkingObservation
    evidence:RestrictedSurfaceEvidence; motion_score:float=0.


class RestrictedParkingEngine:
    """Deterministic source-time parking state machine keyed by track and zone."""
    def __init__(self,config:RestrictedParkingConfig):
        self.config=config; self._memory={}; self.resolver=RestrictedSurfaceResolver(RestrictedSurfaceMode(config.surface_mode),
            config.restricted_surface_classes,config.minimum_surface_confidence)

    def update(self,observations,*,source_time_s:float,width:int,height:int,semantic_result=None)->tuple[ParkingEvent,...]:
        if source_time_s<0: raise ValueError("source_time_s must be non-negative")
        events=[]; seen=set()
        for observation in sorted(observations,key=lambda x:x.track_id):
            if observation.detection_confidence<self.config.minimum_detection_confidence: continue
            anchor=((observation.bbox.x1+observation.bbox.x2)/2/max(width,1),observation.bbox.y2/max(height,1))
            for zone in self.config.zones:
                if not zone.enabled: continue
                key=(observation.track_id,zone.zone_id); memory=self._memory.get(key)
                restricted=self.resolver.resolve(zone,width,height,semantic_result)
                evidence=self.resolver.evaluate(observation.bbox,restricted,semantic_result,width=width,height=height,
                    ground_contact_fraction=self.config.ground_contact_fraction,minimum_overlap=self.config.minimum_restricted_overlap)
                if evidence.uncertain:
                    if memory is None: self._memory[key]=_Memory(ParkingState.UNCERTAIN,anchor,source_time_s,source_time_s,None,None,observation,evidence)
                    continue
                seen.add(key)
                if not evidence.occupied:
                    if memory and memory.state is ParkingState.ILLEGALLY_PARKED: events.append(self._event(zone.zone_id,memory,ParkingState.CLEARED,source_time_s))
                    self._memory.pop(key,None); continue
                if memory is None:
                    memory=_Memory(ParkingState.ON_RESTRICTED_SURFACE,anchor,source_time_s,source_time_s,None,None,observation,evidence); self._memory[key]=memory
                    continue
                elapsed=source_time_s-memory.last_seen_s
                motion=hypot(anchor[0]-memory.anchor[0],anchor[1]-memory.anchor[1])/(elapsed if elapsed>0 else 1)
                was_illegal=memory.state is ParkingState.ILLEGALLY_PARKED
                if self.config.footpath_enforcement_mode:
                    occupancy_seconds=source_time_s-memory.entered_s
                    memory.stationary_since=memory.entered_s
                    if occupancy_seconds>=self.config.violation_min_seconds:
                        memory.state=ParkingState.ILLEGALLY_PARKED
                        if memory.violation_s is None: memory.violation_s=source_time_s
                    else: memory.state=ParkingState.STATIONARY_PENDING
                elif motion<=self.config.stationary_motion_threshold:
                    if memory.stationary_since is None: memory.stationary_since=memory.last_seen_s
                    stationary_seconds=source_time_s-memory.stationary_since
                    if stationary_seconds>=self.config.violation_min_seconds and stationary_seconds>=self.config.stationary_min_seconds:
                        memory.state=ParkingState.ILLEGALLY_PARKED
                        if memory.violation_s is None: memory.violation_s=source_time_s
                    elif stationary_seconds>=self.config.stationary_min_seconds: memory.state=ParkingState.STATIONARY_PENDING
                    else: memory.state=ParkingState.ON_RESTRICTED_SURFACE
                else:
                    if was_illegal: events.append(self._event(zone.zone_id,memory,ParkingState.CLEARED,source_time_s))
                    memory.state=ParkingState.ON_RESTRICTED_SURFACE; memory.stationary_since=None; memory.violation_s=None
                memory.anchor=anchor; memory.last_seen_s=source_time_s; memory.observation=observation; memory.evidence=evidence; memory.motion_score=motion
                if memory.state is ParkingState.ILLEGALLY_PARKED: events.append(self._event(zone.zone_id,memory,memory.state,source_time_s))
        for key,memory in list(self._memory.items()):
            if key not in seen and source_time_s-memory.last_seen_s>self.config.track_grace_seconds:
                if memory.state is ParkingState.ILLEGALLY_PARKED: events.append(self._event(key[1],memory,ParkingState.CLEARED,source_time_s))
                del self._memory[key]
        return tuple(events)

    def state(self,track_id,zone_id):
        memory=self._memory.get((track_id,zone_id)); return memory.state if memory else ParkingState.OUTSIDE

    def stationary_seconds(self,track_id,zone_id,source_time_s):
        memory=self._memory.get((track_id,zone_id))
        return max(0.,source_time_s-memory.stationary_since) if memory and memory.stationary_since is not None else 0.

    def evidence(self,track_id,zone_id):
        memory=self._memory.get((track_id,zone_id)); return memory.evidence if memory else None

    def motion_score(self,track_id,zone_id):
        memory=self._memory.get((track_id,zone_id)); return memory.motion_score if memory else 0.

    def _event(self,zone_id,memory,state,now):
        observation=memory.observation
        confidence="HIGH" if observation.detection_confidence>=.7 and (observation.class_confidence or 0)>=.55 and memory.evidence.surface_confidence>=self.config.minimum_surface_confidence else "MEDIUM"
        reasons=list(memory.evidence.reason_codes)
        if state is ParkingState.ILLEGALLY_PARKED:
            reasons.extend(("FOOTPATH_OCCUPANCY_CONFIRMED","OCCUPANCY_DWELL_EXCEEDED") if self.config.footpath_enforcement_mode
                else ("STATIONARY_CONFIRMED","VIOLATION_DWELL_EXCEEDED"))
        elif state is ParkingState.CLEARED: reasons.append("LEFT_RESTRICTED_SURFACE")
        return ParkingEvent(self.config.camera_id,zone_id,observation.track_id,state,now,observation.vehicle_class,
            observation.raw_vehicle_class,observation.class_confidence,observation.detection_confidence,memory.entered_s,
            memory.stationary_since,memory.violation_s,max(0.,now-memory.stationary_since) if memory.stationary_since is not None else 0.,
            memory.evidence.surface_class,memory.evidence.raw_surface_class,memory.evidence.surface_confidence,
            memory.evidence.restricted_overlap,self.config.ground_contact_fraction,memory.motion_score,self.config.surface_mode,
            "RPZ-2.0",confidence,tuple(dict.fromkeys(reasons)))
