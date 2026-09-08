import {API_BASE_URL} from "./anprService";

export type ParkingAlertState = "ILLEGALLY_PARKED" | "STATIONARY_PENDING" | "UNCERTAIN" | "CLEARED";
export type RestrictedParkingAlert = {
  alert_id: string; alert_type: "RESTRICTED_PARKING"; status: string; parking_state?: ParkingAlertState;
  camera_id: string; zone_id?: string | null; track_id?: number | null; vehicle_class?: string | null;
  stationary_duration_s?: number | null; restricted_overlap?: number | null; surface_class?: string | null;
  surface_confidence?: number | null; decision_confidence?: string | number | null;
  confirmed_at?: number | null; first_seen_at?: number | null; cleared_at?: number | null;
};

export function normalizeParkingAlert(value: Partial<RestrictedParkingAlert>): RestrictedParkingAlert {
  return {alert_id:value.alert_id||"UNKNOWN",alert_type:"RESTRICTED_PARKING",status:value.status||"UNKNOWN",
    parking_state:value.parking_state,camera_id:value.camera_id||"UNKNOWN",zone_id:value.zone_id??null,
    track_id:value.track_id??null,vehicle_class:value.vehicle_class??null,
    stationary_duration_s:value.stationary_duration_s??null,restricted_overlap:value.restricted_overlap??null,
    surface_class:value.surface_class??null,surface_confidence:value.surface_confidence??null,
    decision_confidence:value.decision_confidence??null,confirmed_at:value.confirmed_at??null,
    first_seen_at:value.first_seen_at??null,cleared_at:value.cleared_at??null};
}

export async function getRestrictedParkingAlerts(): Promise<RestrictedParkingAlert[]> {
  const response=await fetch(`${API_BASE_URL}/api/alerts?alert_type=RESTRICTED_PARKING`,{signal:AbortSignal.timeout(5000)});
  if(!response.ok) throw new Error(`HTTP ${response.status}`);
  return ((await response.json()) as Partial<RestrictedParkingAlert>[]).map(normalizeParkingAlert);
}
