from backend.alerts.models import AlertSeverity, AlertType

SEVERITY_BY_TYPE = {
    AlertType.RESTRICTED_PARKING: AlertSeverity.HIGH,
    AlertType.CONGESTION: AlertSeverity.HIGH,
    AlertType.SEVERE_DELAY: AlertSeverity.CRITICAL,
}

def severity_for(alert_type: AlertType) -> AlertSeverity:
    return SEVERITY_BY_TYPE.get(alert_type, AlertSeverity.MEDIUM)

