import {API_BASE_URL} from "./anprService";

export type AnalyticsWindow = {startTime: string; endTime: string};
export type CongestionStatus = "INSUFFICIENT_DATA" | "FREE_FLOW" | "MODERATE" | "CONGESTED" | "SEVERE";

export type NetworkSummary = {
  window_start: string; window_end: string; total_sightings: number; unique_vehicles: number;
  active_cameras: number; inter_camera_movements: number; busiest_camera: string | null;
  busiest_camera_count: number; busiest_segment: string | null; busiest_segment_count: number;
  average_valid_speed_kmh: number | null; median_valid_speed_kmh: number | null;
  free_flow_segments: number; moderate_segments: number; congested_segments: number; severe_segments: number;
};
export type CameraVolume = {camera_id: string; vehicle_count: number; unique_vehicle_count: number; vehicles_per_hour: number; window_start: string; window_end: string};
export type ODFlow = {origin_camera: string; destination_camera: string; vehicle_count: number};
export type SegmentAnalytics = {source_camera: string; target_camera: string; vehicle_count: number; mean_travel_time_s: number; median_travel_time_s: number; mean_speed_kmh: number; median_speed_kmh: number};
export type CongestionResult = {source_camera: string; target_camera: string; sample_count: number; baseline_travel_time_s: number; observed_median_travel_time_s: number | null; travel_time_ratio: number | null; classification: CongestionStatus};
export type HeatmapPoint = {camera_id: string; latitude: number; longitude: number; vehicle_count: number; normalized_intensity: number};
export type RouteDensity = {source: string; target: string; vehicle_count: number; normalized_density: number};
export type AnalyticsSnapshot = {summary: NetworkSummary; cameraVolume: CameraVolume[]; od: ODFlow[]; segments: SegmentAnalytics[]; congestion: CongestionResult[]; heatmap: HeatmapPoint[]; routes: RouteDensity[]};

function query(window: AnalyticsWindow) {
  return new URLSearchParams({start_time: window.startTime, end_time: window.endTime}).toString();
}

async function getAnalytics<T>(path: string, window: AnalyticsWindow): Promise<T> {
  const response = await fetch(`${API_BASE_URL}/api/analytics/${path}?${query(window)}`, {signal: AbortSignal.timeout(7000)});
  if (!response.ok) throw new Error(`Analytics API returned HTTP ${response.status}`);
  return response.json() as Promise<T>;
}

export const getAnalyticsSummary = (window: AnalyticsWindow) => getAnalytics<NetworkSummary>("summary", window);
export const getCameraVolume = (window: AnalyticsWindow) => getAnalytics<CameraVolume[]>("camera-volume", window);
export const getODFlows = (window: AnalyticsWindow) => getAnalytics<ODFlow[]>("od", window);
export const getSegmentAnalytics = (window: AnalyticsWindow) => getAnalytics<SegmentAnalytics[]>("segments", window);
export const getCongestion = (window: AnalyticsWindow) => getAnalytics<CongestionResult[]>("congestion", window);
export const getHeatmapData = (window: AnalyticsWindow) => getAnalytics<HeatmapPoint[]>("heatmap", window);
export const getRouteDensity = (window: AnalyticsWindow) => getAnalytics<RouteDensity[]>("routes", window);

export async function getAnalyticsSnapshot(window: AnalyticsWindow): Promise<AnalyticsSnapshot> {
  const [summary, cameraVolume, od, segments, congestion, heatmap, routes] = await Promise.all([
    getAnalyticsSummary(window), getCameraVolume(window), getODFlows(window), getSegmentAnalytics(window),
    getCongestion(window), getHeatmapData(window), getRouteDensity(window),
  ]);
  return {summary, cameraVolume, od, segments, congestion, heatmap, routes};
}
