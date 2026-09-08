from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AlertType(str, Enum):
    RESTRICTED_PARKING="RESTRICTED_PARKING"; BLACKLISTED_VEHICLE="BLACKLISTED_VEHICLE"
    RESTRICTED_ZONE="RESTRICTED_ZONE"; ROUTE_ANOMALY="ROUTE_ANOMALY"; CONGESTION="CONGESTION"
    SEVERE_DELAY="SEVERE_DELAY"; BLOCKAGE="BLOCKAGE"; POSSIBLE_ACCIDENT="POSSIBLE_ACCIDENT"
    AMBULANCE_PRIORITY="AMBULANCE_PRIORITY"; FIRE_VEHICLE_PRIORITY="FIRE_VEHICLE_PRIORITY"


class AlertSeverity(str, Enum): LOW="LOW"; MEDIUM="MEDIUM"; HIGH="HIGH"; CRITICAL="CRITICAL"
class AlertStatus(str, Enum): CREATED="CREATED"; ACTIVE="ACTIVE"; UPDATED="UPDATED"; CLEARED="CLEARED"


@dataclass(frozen=True, slots=True)
class Alert:
    alert_id:str; alert_type:AlertType; severity:AlertSeverity; status:AlertStatus
    camera_id:str; zone_id:str|None; track_id:int|None; vehicle_class:str|None
    raw_vehicle_class:str|None; plate:str|None; plate_status:str|None
    started_at:float; last_updated_at:float; cleared_at:float|None; confidence:float
    metadata:dict[str,Any]=field(default_factory=dict)

    @property
    def lifecycle_key(self): return self.camera_id,self.zone_id,self.track_id,self.alert_type

    def to_dict(self):
        payload={"alert_id":self.alert_id,"alert_type":self.alert_type.value,"severity":self.severity.value,
            "status":self.status.value,"camera_id":self.camera_id,"zone_id":self.zone_id,"track_id":self.track_id,
            "vehicle_class":self.vehicle_class,"raw_vehicle_class":self.raw_vehicle_class,"plate":self.plate,
            "plate_text":self.plate,
            "plate_status":self.plate_status,"started_at":self.started_at,"last_updated_at":self.last_updated_at,
            "cleared_at":self.cleared_at,"confidence":self.confidence,"metadata":dict(self.metadata)}
        if self.alert_type is AlertType.RESTRICTED_PARKING:
            metadata=self.metadata
            payload.update({"parking_state":"CLEARED" if self.status is AlertStatus.CLEARED else "ILLEGALLY_PARKED",
                "vehicle_class_confidence":metadata.get("class_confidence"),
                "detection_confidence":self.confidence,"motion_score":metadata.get("motion_score"),
                "stationary_duration_s":metadata.get("stationary_seconds"),
                "ground_contact_fraction":metadata.get("ground_contact_fraction"),
                "restricted_overlap":metadata.get("restricted_overlap"),"surface_class":metadata.get("surface_class"),
                "surface_confidence":metadata.get("surface_confidence"),"policy_mode":metadata.get("policy_mode"),
                "decision_confidence":metadata.get("decision_confidence"),
                "reason_codes":list(metadata.get("decision_reason_codes",())),
                "first_seen_at":metadata.get("entered_zone_at"),
                "confirmed_at":metadata.get("violation_started_at"),"rule_version":metadata.get("rule_version")})
        elif self.alert_type is AlertType.BLACKLISTED_VEHICLE:
            metadata=self.metadata
            payload.update({key:metadata.get(key) for key in ("plate_text","plate_confidence","vehicle_class",
                "watchlist_id","watchlist_reason","watchlist_severity","watchlist_source","demo_watchlist_entry",
                "detected_at","rule_version","reason_codes")})
        elif self.alert_type is AlertType.RESTRICTED_ZONE:
            metadata=self.metadata
            payload.update({key:metadata.get(key) for key in ("zone_label","vehicle_class_confidence",
                "plate_text","plate_confidence","entry_time","confirmed_time","cleared_time","inside_duration_s",
                "authorization_status","authorization_reason","rule_version","reason_codes","decision_confidence")})
        elif self.alert_type in (AlertType.CONGESTION, AlertType.SEVERE_DELAY):
            metadata=self.metadata
            payload.update({key:metadata.get(key) for key in (
                "segment_id","origin_camera","destination_camera",
                "congestion_state","delay_ratio",
                "observed_travel_time_s","baseline_travel_time_s",
                "sample_count","first_qualifying_at","confirmed_at","cleared_at",
                "reason_codes","rule_version")})
        return payload

