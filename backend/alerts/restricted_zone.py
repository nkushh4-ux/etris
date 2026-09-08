from dataclasses import replace

from backend.alerts.models import Alert, AlertStatus, AlertType
from backend.alerts.repository import AlertRepository
from backend.alerts.rules import severity_for
from cv.alerts.restricted_zone import RestrictedZoneState


class RestrictedZoneAlertHandler:
    def __init__(self,repository:AlertRepository): self.repository=repository; self._sequence=0
    def handle(self,event):
        key=(event.camera_id,event.zone_id,event.track_id,AlertType.RESTRICTED_ZONE)
        current=self.repository.get_active_by_key(key)
        if event.state is RestrictedZoneState.VIOLATION:
            metadata={"zone_label":event.zone_label,"vehicle_class_confidence":event.vehicle_class_confidence,
                "plate_text":event.plate_text,"plate_confidence":event.plate_confidence,"entry_time":event.entry_time,
                "confirmed_time":event.confirmed_time,"cleared_time":event.cleared_time,
                "inside_duration_s":event.inside_duration_s,"authorization_status":event.authorization_status,
                "authorization_reason":event.authorization_reason,"rule_version":event.rule_version,
                "reason_codes":list(event.reason_codes),"decision_confidence":event.decision_confidence}
            if current is None:
                self._sequence+=1; current=Alert(f"ALT-RZ-{self._sequence:05d}",AlertType.RESTRICTED_ZONE,
                    severity_for(AlertType.RESTRICTED_ZONE),AlertStatus.ACTIVE,event.camera_id,event.zone_id,event.track_id,
                    event.vehicle_class,None,event.plate_text,"ACCEPTED" if event.plate_text else None,event.entry_time,
                    event.confirmed_time or event.entry_time,None,event.plate_confidence or 0.,metadata)
            else: current=replace(current,status=AlertStatus.UPDATED,last_updated_at=event.inside_duration_s+event.entry_time,metadata=metadata)
            self.repository.save(current); return current
        if event.state is RestrictedZoneState.CLEARED and current is not None:
            current=replace(current,status=AlertStatus.CLEARED,last_updated_at=event.cleared_time or current.last_updated_at,
                cleared_at=event.cleared_time); self.repository.save(current); return current
        return None
