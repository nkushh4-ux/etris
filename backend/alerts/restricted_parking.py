from dataclasses import replace

from backend.alerts.models import Alert, AlertStatus, AlertType
from backend.alerts.repository import AlertRepository
from backend.alerts.rules import severity_for
from cv.alerts.restricted_parking import ParkingEvent, ParkingState


class RestrictedParkingAlertHandler:
    def __init__(self,repository:AlertRepository): self.repository=repository; self._sequence=0
    def handle(self,event:ParkingEvent)->Alert|None:
        key=(event.camera_id,event.zone_id,event.track_id,AlertType.RESTRICTED_PARKING)
        current=self.repository.get_active_by_key(key)
        if event.state is ParkingState.ILLEGALLY_PARKED:
            metadata={"entered_zone_at":event.entered_zone_at,"stationary_since":event.stationary_since,
                "violation_started_at":event.violation_started_at,"stationary_seconds":event.stationary_seconds,
                "surface_class":event.surface_class,"raw_surface_class":event.raw_surface_class,
                "surface_confidence":event.surface_confidence,"restricted_overlap":event.restricted_overlap,
                "ground_contact_fraction":event.ground_contact_fraction,"motion_score":event.motion_score,
                "class_confidence":event.class_confidence,"policy_mode":event.policy_mode,"rule_version":event.rule_version,
                "decision_confidence":event.decision_confidence,"decision_reason_codes":list(event.decision_reason_codes)}
            if current is None:
                self._sequence+=1
                current=Alert(f"ALT-RPZ-{self._sequence:05d}",AlertType.RESTRICTED_PARKING,
                    severity_for(AlertType.RESTRICTED_PARKING),AlertStatus.ACTIVE,event.camera_id,event.zone_id,
                    event.track_id,event.vehicle_class,event.raw_vehicle_class,None,None,event.source_time_s,
                    event.source_time_s,None,event.detection_confidence,metadata)
            else:
                current=replace(current,status=AlertStatus.UPDATED,last_updated_at=event.source_time_s,
                    vehicle_class=event.vehicle_class,raw_vehicle_class=event.raw_vehicle_class,
                    confidence=event.detection_confidence,metadata=metadata)
            self.repository.save(current); return current
        if event.state is ParkingState.CLEARED and current is not None:
            current=replace(current,status=AlertStatus.CLEARED,last_updated_at=event.source_time_s,cleared_at=event.source_time_s)
            self.repository.save(current); return current
        return None
