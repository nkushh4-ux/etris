import {API_BASE_URL} from "./anprService";

export type SignalDecision = {
  intersection_id: string;
  approach_id: string;
  pressure: number;
  effective_priority: number;
  green_duration_s: number;
  fairness_applied: boolean;
  reason_codes: string[];
};

export type SignalCyclePlan = {
  intersection_id: string;
  generated_at: string;
  ordered_phases: SignalDecision[];
  cycle_duration_s: number;
  recommendation_only: boolean;
};

export type DemoSignalResponse = {
  data_source: "DEMO_SYNTHETIC";
  scenario: "BASELINE" | "EAST_DEMAND_SURGE";
  plan: SignalCyclePlan;
};

export type RealTrafficApproach = {
  approach_id: string; label: string; vehicle_count: number; queue_length: number;
  occupancy: number; arrival_rate_vpm: number; average_waiting_time_s: number;
  heavy_vehicle_count: number; queued_track_ids: number[]; controller: SignalDecision;
};
export type RealTrafficEvent = {source:"REAL_VIDEO_ESTIMATE"; video_time_s:number; frame_index:number;
  approaches:RealTrafficApproach[]; phase_order:string[]; cycle_duration_s:number; limitations:string};

async function getDemo(path: string): Promise<DemoSignalResponse> {
  const response = await fetch(`${API_BASE_URL}/api/signal-control/${path}`, {signal: AbortSignal.timeout(7000)});
  if (!response.ok) throw new Error(`Signal-control API returned HTTP ${response.status}`);
  return response.json() as Promise<DemoSignalResponse>;
}

export const getBaselineSignalPlan = () => getDemo("demo");
export const getEastSurgeSignalPlan = () => getDemo("demo/change");
export const getTrafficDemoVideoURL = () => `${API_BASE_URL}/api/signal-control/traffic-demo/video`;
export async function getRealTrafficEvents(): Promise<RealTrafficEvent[]> {
  const response=await fetch(`${API_BASE_URL}/api/signal-control/traffic-demo/events`,{signal:AbortSignal.timeout(7000)});
  if (!response.ok) throw new Error(`Traffic-state events returned HTTP ${response.status}`);
  return response.json() as Promise<RealTrafficEvent[]>;
}
