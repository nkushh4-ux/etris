export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

export type DatabaseStatus = {mode: "memory" | "postgresql"; configured: boolean; connected: boolean; last_error: string | null};
export type ANPRStatus = {backend_online: boolean; mode: "LIVE_ANALYSIS" | "RECORDED_ANALYSIS"; processing_rate: string; running: boolean; frame: number; fps: number; source: string; camera_id: string; pipeline_status: string; error: string | null; playback_state: "PLAYING" | "PAUSED" | "ENDED"; current_frame: number; total_frames: number; current_time_s: number; duration_s: number; source_fps: number; playback_fps: number; inference_fps: number; latest_analyzed_frame: number; database?: DatabaseStatus};
export type ANPREvent = {plate: string; track_id: number; status: "ACCEPTED" | "LOW_CONFIDENCE" | "UNKNOWN" | "PROVISIONAL" | "LIVE_CONSENSUS"; confidence: number; camera_id: string; timestamp: string; frame_id: number; video_time_s?: number | null};
export type GroupedANPREvent = {main: ANPREvent; evidence: ANPREvent[]};

const AUTHORITY: Record<ANPREvent["status"], number> = {
  UNKNOWN: 0, LOW_CONFIDENCE: 0, PROVISIONAL: 1, LIVE_CONSENSUS: 2, ACCEPTED: 3,
};

export function groupEventsByTrack(events: ANPREvent[], showUnknowns = false): GroupedANPREvent[] {
  const groups = new Map<string, ANPREvent[]>();
  for (const event of events) {
    if (!showUnknowns && (event.status === "UNKNOWN" || event.plate === "UNKNOWN")) continue;
    const key = `${event.camera_id}:${event.track_id}`;
    groups.set(key, [...(groups.get(key) || []), event]);
  }
  return [...groups.values()].map(evidence => ({
    evidence,
    main: [...evidence].sort((a, b) => AUTHORITY[b.status] - AUTHORITY[a.status]
      || b.frame_id - a.frame_id || b.timestamp.localeCompare(a.timestamp))[0],
  })).sort((a, b) => b.main.frame_id - a.main.frame_id
    || a.main.camera_id.localeCompare(b.main.camera_id) || a.main.track_id - b.main.track_id);
}

async function getJSON<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {signal: AbortSignal.timeout(5000)});
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json() as Promise<T>;
}

async function postJSON<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {method: "POST"});
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json() as Promise<T>;
}

async function postBody<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body), signal: AbortSignal.timeout(5000)});
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.json() as Promise<T>;
}

export const getANPRStreamURL = (cacheKey?: number) =>
  `${API_BASE_URL}/api/anpr/stream${cacheKey === undefined ? "" : `?t=${cacheKey}`}`;
export type ANPRMode = "LIVE_ANALYSIS" | "RECORDED_ANALYSIS";
export const RECORDED_PLAYBACK_RATES = [1, 1.25, 1.5] as const;
export const getANPRStatus = (mode: ANPRMode = "LIVE_ANALYSIS") => getJSON<ANPRStatus>(`/api/anpr/status?mode=${mode}`);
export const getRecentANPREvents = () => getJSON<ANPREvent[]>("/api/anpr/events");
export const getCurrentRecognition = () => getJSON<ANPREvent | null>("/api/anpr/recognition");
export const getRecordedANPREvents = () => getJSON<ANPREvent[]>("/api/anpr/recorded-events");
export const getRecordedVideoURL = () => `${API_BASE_URL}/api/anpr/recorded-video`;
export const eventsAtPlaybackTime = (events: ANPREvent[], timeS: number) =>
  events.filter(event => event.video_time_s !== null && event.video_time_s !== undefined && event.video_time_s <= timeS);
export const playANPR = () => postJSON<ANPRStatus>("/api/anpr/play");
export const pauseANPR = () => postJSON<ANPRStatus>("/api/anpr/pause");
export const restartANPR = () => postJSON<ANPRStatus>("/api/anpr/restart");
export const setLiveProcessingRate = (rate: "AUTO" | "1.0x") => postBody<ANPRStatus>("/api/anpr/processing-rate", {rate});
export const isValidANPRStatus = (value: unknown): value is ANPRStatus => {
  const status = value as Partial<ANPRStatus> | null;
  return !!status && status.backend_online === true && typeof status.camera_id === "string" && typeof status.current_frame === "number" && typeof status.inference_fps === "number";
};
