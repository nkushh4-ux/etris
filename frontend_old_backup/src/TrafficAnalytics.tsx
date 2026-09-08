import {useCallback, useEffect, useMemo, useState} from "react";
import type {ReactNode} from "react";
import {AnalyticsSnapshot, CongestionStatus, getAnalyticsSnapshot} from "./services/analyticsService";

type WindowHours = 1 | 6 | 24;

export default function TrafficAnalytics() {
  const [hours, setHours] = useState<WindowHours>(24);
  const [snapshot, setSnapshot] = useState<AnalyticsSnapshot | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const load = useCallback(async () => {
    setLoading(true); setError("");
    const end = new Date(); const start = new Date(end.getTime() - hours * 3600000);
    try { setSnapshot(await getAnalyticsSnapshot({startTime:start.toISOString(), endTime:end.toISOString()})); }
    catch (reason) {
      setSnapshot(null);
      setError(reason instanceof TypeError || (reason instanceof DOMException && reason.name === "TimeoutError")
        ? "Traffic analytics backend is offline." : reason instanceof Error ? reason.message : "Traffic analytics request failed.");
    } finally { setLoading(false); }
  }, [hours]);

  useEffect(() => { void load(); }, [load]);
  const empty = snapshot?.summary.total_sightings === 0;
  const congested = snapshot ? snapshot.summary.congested_segments + snapshot.summary.severe_segments : 0;
  const cameraMax = Math.max(0, ...(snapshot?.cameraVolume.map(x=>x.vehicle_count) || []));
  const od = useMemo(() => [...(snapshot?.od || [])].sort((a,b)=>b.vehicle_count-a.vehicle_count || a.origin_camera.localeCompare(b.origin_camera)).slice(0,8), [snapshot]);
  const routes = useMemo(() => [...(snapshot?.routes || [])].sort((a,b)=>b.vehicle_count-a.vehicle_count || a.source.localeCompare(b.source)).slice(0,8), [snapshot]);

  return <section className="analyticsView">
    <div className="analyticsToolbar"><div><span className="eyebrow">CITY NETWORK</span><h2>Traffic Analytics</h2></div><div className="windowControl" aria-label="Analytics time window"><span>TIME WINDOW</span>{([1,6,24] as WindowHours[]).map(value=><button key={value} className={hours===value?"active":""} onClick={()=>setHours(value)}>Last {value}h</button>)}<button onClick={()=>void load()}>Refresh</button></div></div>
    {loading && <StatePanel title="Loading traffic analytics…" detail="Reading confirmed sightings from the active repository."/>}
    {!loading && error && <StatePanel title={error} detail="Live ANPR remains available. Retry when the analytics API is reachable." action={()=>void load()}/>} 
    {!loading && !error && empty && <StatePanel title="No confirmed traffic data in this time window." detail="Analytics excludes low-confidence and unknown plate identities." action={()=>void load()}/>} 
    {!loading && !error && snapshot && !empty && <>
      <div className="analyticsKpis">
        <Kpi label="TOTAL SIGHTINGS" value={snapshot.summary.total_sightings}/><Kpi label="UNIQUE VEHICLES" value={snapshot.summary.unique_vehicles}/><Kpi label="ACTIVE CAMERAS" value={snapshot.summary.active_cameras}/><Kpi label="INTER-CAMERA MOVEMENTS" value={snapshot.summary.inter_camera_movements}/><Kpi label="AVERAGE VALID SPEED" value={snapshot.summary.average_valid_speed_kmh == null ? "—" : `${snapshot.summary.average_valid_speed_kmh.toFixed(1)} km/h`}/><Kpi label="CONGESTED SEGMENTS" value={congested}/>
      </div>
      <div className="analyticsGrid">
        <AnalyticsPanel title="CAMERA VOLUME" meta={`Busiest · ${snapshot.summary.busiest_camera || "—"}`} className="wide">
          <div className="barList">{snapshot.cameraVolume.map(camera=><div className={`barRow ${camera.camera_id===snapshot.summary.busiest_camera?"highlight":""}`} key={camera.camera_id}><div><strong>{camera.camera_id}</strong><span>{camera.vehicle_count} sightings · {camera.unique_vehicle_count} vehicles · {camera.vehicles_per_hour.toFixed(1)}/hr</span></div><div className="barTrack"><i style={{width:`${cameraMax ? camera.vehicle_count/cameraMax*100 : 0}%`}}/></div></div>)}</div>
        </AnalyticsPanel>
        <AnalyticsPanel title="CONGESTION" meta="MVP heuristic">
          <div className="congestionList">{snapshot.congestion.map(item=><div className="congestionRow" key={`${item.source_camera}-${item.target_camera}`}><div><strong>{item.source_camera} → {item.target_camera}</strong><span>{item.sample_count} samples · median {formatSeconds(item.observed_median_travel_time_s)} · ratio {item.travel_time_ratio?.toFixed(2) ?? "—"}</span></div><StatusBadge status={item.classification}/></div>)}</div>
        </AnalyticsPanel>
        <AnalyticsPanel title="ORIGIN → DESTINATION" meta="Top movements">
          <div className="rankList">{od.map((item,index)=><div key={`${item.origin_camera}-${item.destination_camera}`}><span>{String(index+1).padStart(2,"0")}</span><strong>{item.origin_camera} → {item.destination_camera}</strong><b>{item.vehicle_count}</b></div>)}</div>
        </AnalyticsPanel>
        <AnalyticsPanel title="SEGMENT ANALYTICS" meta="Valid topology transitions" className="wide">
          <div className="analyticsTable"><table><thead><tr><th>Segment</th><th>Vehicles</th><th>Median travel</th><th>Median speed</th></tr></thead><tbody>{snapshot.segments.map(item=><tr key={`${item.source_camera}-${item.target_camera}`}><td className="routeName">{item.source_camera} → {item.target_camera}</td><td>{item.vehicle_count}</td><td>{formatSeconds(item.median_travel_time_s)}</td><td>{item.median_speed_kmh.toFixed(1)} km/h</td></tr>)}</tbody></table></div>
        </AnalyticsPanel>
        <AnalyticsPanel title="ROUTE DENSITY" meta="Relative to busiest segment">
          <div className="barList compact">{routes.map(item=><div className="barRow" key={`${item.source}-${item.target}`}><div><strong>{item.source} → {item.target}</strong><span>{item.vehicle_count} vehicles · {(item.normalized_density*100).toFixed(0)}%</span></div><div className="barTrack"><i style={{width:`${item.normalized_density*100}%`}}/></div></div>)}</div>
        </AnalyticsPanel>
        <AnalyticsPanel title="CAMERA INTENSITY" meta="Heatmap-ready">
          <div className="intensityGrid">{snapshot.heatmap.map(item=><div key={item.camera_id}><i style={{opacity:.18+.82*item.normalized_intensity}}/><strong>{item.camera_id}</strong><span>{item.vehicle_count} · {(item.normalized_intensity*100).toFixed(0)}%</span></div>)}</div>
        </AnalyticsPanel>
      </div>
      <div className="analyticsFoot">Confirmed sightings only · {new Date(snapshot.summary.window_start).toLocaleString()} — {new Date(snapshot.summary.window_end).toLocaleString()}</div>
    </>}
  </section>;
}

function AnalyticsPanel({title,meta,className="",children}:{title:string;meta:string;className?:string;children:ReactNode}){return <article className={`panel analyticsPanel ${className}`}><div className="panelTitle"><span>{title}</span><small>{meta}</small></div>{children}</article>}
function Kpi({label,value}:{label:string;value:string|number}){return <div className="analyticsKpi"><span>{label}</span><strong>{value}</strong></div>}
function StatePanel({title,detail,action}:{title:string;detail:string;action?:()=>void}){return <div className="analyticsState"><strong>{title}</strong><span>{detail}</span>{action&&<button onClick={action}>Retry</button>}</div>}
function StatusBadge({status}:{status:CongestionStatus}){return <span className={`congestionBadge status-${status.toLowerCase()}`}>{status.replaceAll("_"," ")}</span>}
function formatSeconds(value:number|null){return value == null ? "—" : value < 60 ? `${value.toFixed(0)}s` : `${Math.floor(value/60)}m ${Math.round(value%60)}s`}
