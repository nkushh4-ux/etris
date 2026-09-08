import {useCallback, useEffect, useMemo, useRef, useState} from "react";
import {ANPREvent, ANPRMode, ANPRStatus, eventsAtPlaybackTime, getANPRStatus, getANPRStreamURL, getCurrentRecognition, getRecentANPREvents, getRecordedANPREvents, getRecordedVideoURL, groupEventsByTrack, isValidANPRStatus, pauseANPR, playANPR, RECORDED_PLAYBACK_RATES, restartANPR, setLiveProcessingRate} from "./services/anprService";
import TrafficAnalytics from "./TrafficAnalytics";
import SignalControl from "./SignalControl";
import AlertCentre from "./AlertCentre";
import "./styles.css";

type EventFilter = "MEANINGFUL" | "ACCEPTED" | "LIVE_CONSENSUS" | "PROVISIONAL";

export default function App() {
  const [view, setView] = useState<"ANPR" | "ANALYTICS" | "SIGNAL_CONTROL" | "ALERTS">("ANPR");
  const [mode, setMode] = useState<ANPRMode>("RECORDED_ANALYSIS");
  const [status, setStatus] = useState<ANPRStatus | null>(null);
  const [events, setEvents] = useState<ANPREvent[]>([]);
  const [offline, setOffline] = useState(false);
  const [streamError, setStreamError] = useState(false);
  const [retryKey, setRetryKey] = useState(Date.now());
  const [pending, setPending] = useState(false);
  const [message, setMessage] = useState("");
  const [videoTime, setVideoTime] = useState(0);
  const [videoDuration, setVideoDuration] = useState(0);
  const [recordedPlaying, setRecordedPlaying] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1.25);
  const [liveRecognition, setLiveRecognition] = useState<ANPREvent | null>(null);
  const [presentedLiveRecognition, setPresentedLiveRecognition] = useState<ANPREvent | null>(null);
  const [lastAccepted, setLastAccepted] = useState<ANPREvent | null>(null);
  const lastRecognitionKey = useRef("");
  const [liveRate, setLiveRate] = useState<"AUTO" | "1.0x">("AUTO");
  const [eventFilter, setEventFilter] = useState<EventFilter>("MEANINGFUL");
  const [showUnknowns, setShowUnknowns] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    let active = true;
    let timer = 0;
    let failures = 0;
    const update = async () => {
      try {
        const nextStatus = await getANPRStatus(mode);
        if (!isValidANPRStatus(nextStatus)) throw new Error("Invalid status payload");
        failures = 0;
        if (active) { setStatus(nextStatus); setOffline(false); }
        // Auxiliary data failures do not mean the backend health request failed.
        try {
          const nextEvents = mode === "LIVE_ANALYSIS" ? await getRecentANPREvents() : await getRecordedANPREvents();
          if (active) setEvents(nextEvents);
        } catch { /* retain previous events */ }
        if (mode === "LIVE_ANALYSIS") {
          try { const recognition = await getCurrentRecognition(); if (active) setLiveRecognition(recognition); }
          catch { /* retain previous recognition */ }
        }
      } catch { if (active) { failures += 1; setOffline(true); } }
      if (active) timer = window.setTimeout(update, failures ? Math.min(5000, 1000 * 2 ** failures) : mode === "LIVE_ANALYSIS" ? 1000 : 5000);
    };
    void update();
    return () => { active = false; window.clearTimeout(timer); };
  }, [mode]);

  useEffect(() => {
    if (!streamError) return;
    const timer = window.setTimeout(() => { setStreamError(false); setRetryKey(Date.now()); }, 1500);
    return () => window.clearTimeout(timer);
  }, [streamError]);

  const visibleEvents = useMemo(() => mode === "RECORDED_ANALYSIS"
    ? eventsAtPlaybackTime(events, videoTime) : events, [events, mode, videoTime]);
  const liveRecognitionKey = liveRecognition
    ? `${liveRecognition.camera_id}:${liveRecognition.track_id}:${liveRecognition.frame_id}:${liveRecognition.status}:${liveRecognition.plate}`
    : "";
  useEffect(() => {
    if (!liveRecognition || liveRecognition.plate === "UNKNOWN") {
      setPresentedLiveRecognition(current => current?.status === "ACCEPTED" ? current : null);
      return;
    }
    const key = liveRecognitionKey;
    if (key === lastRecognitionKey.current) return;
    lastRecognitionKey.current = key;
    if (liveRecognition.status === "ACCEPTED") {
      setLastAccepted(liveRecognition);
      setPresentedLiveRecognition(liveRecognition);
      const timer = window.setTimeout(() => setPresentedLiveRecognition(current =>
        current?.status === "ACCEPTED" && current.frame_id === liveRecognition.frame_id ? null : current), 3000);
      return () => window.clearTimeout(timer);
    } else {
      setPresentedLiveRecognition(current => current?.status === "ACCEPTED" ? current : liveRecognition);
    }
  }, [liveRecognitionKey]);
  const currentRecognition = useMemo(() => {
    if (mode === "LIVE_ANALYSIS") return presentedLiveRecognition;
    const meaningful = visibleEvents.filter(event => event.plate !== "UNKNOWN");
    const active = meaningful.filter(event => videoTime - (event.video_time_s || 0) <= 3);
    return groupEventsByTrack(active)[0]?.main || null;
  }, [mode, presentedLiveRecognition, visibleEvents, videoTime]);
  const groupedEvents = useMemo(() => groupEventsByTrack(visibleEvents, showUnknowns).filter(group =>
    eventFilter === "MEANINGFUL" || group.main.status === eventFilter), [visibleEvents, eventFilter, showUnknowns]);
  const acceptedCount = useMemo(() => visibleEvents.filter(event => event.status === "ACCEPTED").length, [visibleEvents]);

  const toggle = useCallback(async () => {
    setPending(true); setMessage("");
    try {
      if (mode === "RECORDED_ANALYSIS") {
        const video = videoRef.current;
        if (!video) return;
        if (video.paused) await video.play(); else video.pause();
      } else {
        setStatus(status?.playback_state === "PLAYING" ? await pauseANPR() : await playANPR());
      }
    } catch { setMessage("Playback control unavailable"); }
    finally { setPending(false); }
  }, [mode, status?.playback_state]);

  const restart = async () => {
    setPending(true); setMessage("");
    try {
      if (mode === "RECORDED_ANALYSIS") {
        const video = videoRef.current; if (video) { video.currentTime = 0; await video.play(); }
      } else setStatus(await restartANPR());
    } catch { setMessage("Restart unavailable"); }
    finally { setPending(false); }
  };

  const changeLiveRate = async (rate: "AUTO" | "1.0x") => {
    setPending(true); setMessage("");
    try { setStatus(await setLiveProcessingRate(rate)); setLiveRate(rate); }
    catch { setMessage("Processing-rate control unavailable"); }
    finally { setPending(false); }
  };

  const currentTime = mode === "RECORDED_ANALYSIS" ? videoTime : status?.current_time_s || 0;
  const duration = mode === "RECORDED_ANALYSIS" ? videoDuration : status?.duration_s || 0;
  const currentFrame = mode === "RECORDED_ANALYSIS" ? Math.floor(videoTime * (status?.source_fps || 0)) : status?.current_frame || 0;
  const isPlaying = mode === "RECORDED_ANALYSIS" ? recordedPlaying : status?.playback_state === "PLAYING";

  return <main>
    <header><div><span className="eyebrow">ETRIS ULTIMATE AI</span><h1>{view === "ANPR" ? "ANPR Analysis" : view === "ANALYTICS" ? "Traffic Analytics" : view === "SIGNAL_CONTROL" ? "Signal Control" : "Alert Centre"}</h1></div><div className="healthStack"><div className={`health ${offline ? "off" : ""}`}><i />{offline ? "BACKEND OFFLINE" : status ? "BACKEND ONLINE" : "CONNECTING"}</div>{status?.database && <small>DATABASE · {status.database.mode === "memory" ? "MEMORY MODE" : status.database.connected ? "CONNECTED" : "DEGRADED"}</small>}</div></header>
    <nav className="primaryNav" aria-label="Primary navigation"><button className={view === "ANPR" ? "active" : ""} onClick={()=>setView("ANPR")}>LIVE ANPR</button><button className={view === "ANALYTICS" ? "active" : ""} onClick={()=>setView("ANALYTICS")}>TRAFFIC ANALYTICS</button><button className={view === "SIGNAL_CONTROL" ? "active" : ""} onClick={()=>setView("SIGNAL_CONTROL")}>SIGNAL CONTROL</button><button className={view === "ALERTS" ? "active" : ""} onClick={()=>setView("ALERTS")}>ALERT CENTRE</button></nav>
    {view === "ANALYTICS" ? <TrafficAnalytics/> : view === "SIGNAL_CONTROL" ? <SignalControl/> : view === "ALERTS" ? <AlertCentre/> : <>
    <div className="modeSelector">
      <button className={mode === "RECORDED_ANALYSIS" ? "active" : ""} onClick={() => setMode("RECORDED_ANALYSIS")}><strong>RECORDED ANALYSIS</strong><span>Preprocessed real ANPR output</span></button>
      <button className={mode === "LIVE_ANALYSIS" ? "active" : ""} onClick={() => setMode("LIVE_ANALYSIS")}><strong>LIVE AI</strong><span>Real-time model execution</span></button>
    </div>
    <section className="metrics">
      <Metric label="SYSTEM" value={mode === "RECORDED_ANALYSIS" ? "PREPROCESSED" : status?.playback_state || "STARTING"}/>
      <Metric label="CURRENT FRAME" value={`${Math.max(0,currentFrame)} / ${status?.total_frames || 0}`}/>
      <Metric label={mode === "LIVE_ANALYSIS" ? "AI FPS / VIDEO FPS" : "PLAYBACK / VIDEO FPS"} value={mode === "LIVE_ANALYSIS" ? `${(status?.inference_fps || 0).toFixed(1)} / ${(status?.source_fps || 0).toFixed(1)}` : `${playbackRate.toFixed(2)}× / ${(status?.source_fps || 0).toFixed(1)}`}/>
      <Metric label="CAMERA" value={status?.camera_id || "CAM-DEMO-01"}/>
    </section>
    <section className="workspace">
      <article className="panel stream"><div className="panelTitle"><span>{mode === "LIVE_ANALYSIS" ? "LIVE AI FEED" : "RECORDED ANALYSIS"}</span><small>{status?.source || "annotated_traffic.mp4"}</small></div><div className="viewport">
        {mode === "RECORDED_ANALYSIS" ? <video ref={videoRef} src={getRecordedVideoURL()} preload="metadata" onTimeUpdate={e => setVideoTime(e.currentTarget.currentTime)} onLoadedMetadata={e => {setVideoDuration(e.currentTarget.duration);e.currentTarget.playbackRate=playbackRate;}} onPlay={() => setRecordedPlaying(true)} onPause={() => setRecordedPlaying(false)} onError={() => setMessage("Recorded video unavailable")} /> : <img key={retryKey} src={getANPRStreamURL(retryKey)} alt="Sequential live ANPR stream" onError={() => setStreamError(true)}/>} 
        {streamError && mode === "LIVE_ANALYSIS" && <div className="streamOverlay"><strong>Stream reconnecting...</strong><button onClick={() => {setStreamError(false);setRetryKey(Date.now());}}>Retry now</button></div>}
      </div><div className="controls"><button disabled={pending || offline} onClick={() => void restart()}>Restart</button><button className="primary" disabled={pending || offline} onClick={() => void toggle()}>{isPlaying ? "Pause" : "Play"}</button>{mode === "RECORDED_ANALYSIS" ? <div className="speedControl">{RECORDED_PLAYBACK_RATES.map(rate=><button className={playbackRate===rate?"selected":""} key={rate} onClick={()=>{setPlaybackRate(rate);if(videoRef.current)videoRef.current.playbackRate=rate;}}>{rate.toFixed(rate===1?1:2)}x</button>)}</div> : <div className="liveRate"><span>LIVE PROCESSING RATE</span><div className="speedControl">{(["AUTO","1.0x"] as const).map(rate=><button className={liveRate===rate?"selected":""} key={rate} disabled={pending} onClick={()=>void changeLiveRate(rate)}>{rate}</button>)}</div>{liveRate==="1.0x" && (status?.inference_fps||0)<(status?.source_fps||0) && <small>Hardware-limited: running at actual AI speed</small>}</div>}<div className="progressBlock"><input type="range" min="0" max={duration || 0} step="0.05" value={currentTime} disabled={mode !== "RECORDED_ANALYSIS"} onChange={e => {if(videoRef.current) videoRef.current.currentTime=Number(e.target.value);}}/><div className="progressMeta"><span>{formatTime(currentTime)} / {formatTime(duration)}</span><span>Frame {Math.max(0,currentFrame)} / {status?.total_frames || 0}</span></div></div>{message && <small className="controlError">{message}</small>}</div></article>
      <aside className="sideRail">
        <article className={`panel recognition authority-${currentRecognition?.status.toLowerCase() || "waiting"}`}><div className="recognitionTitle">CURRENT RECOGNITION</div>{currentRecognition && currentRecognition.plate !== "UNKNOWN" ? <><strong>{currentRecognition.plate}</strong><div className="recognitionStatus"><b className={`badge ${currentRecognition.status.toLowerCase()}`}>{statusLabel(currentRecognition.status)}</b></div><dl><div><dt>Confidence</dt><dd>{(currentRecognition.confidence*100).toFixed(2)}%</dd></div><div><dt>Track</dt><dd>#{currentRecognition.track_id}</dd></div><div><dt>Camera</dt><dd>{currentRecognition.camera_id}</dd></div></dl></> : <div className="recognitionEmpty">Waiting for plate...</div>}</article>
        {mode === "LIVE_ANALYSIS" && !currentRecognition && lastAccepted && <article className="panel lastAccepted"><div className="recognitionTitle">LAST ACCEPTED · NOT CURRENT</div><strong>{lastAccepted.plate}</strong><span>{(lastAccepted.confidence*100).toFixed(2)}% · {lastAccepted.camera_id} · Track #{lastAccepted.track_id}</span></article>}
        <article className="panel events"><div className="panelTitle"><span>RECENT MEANINGFUL EVENTS</span><small>Accepted: {acceptedCount}</small></div><div className="eventFilters"><div>{([['MEANINGFUL','All Meaningful'],['ACCEPTED','Accepted'],['LIVE_CONSENSUS','Consensus'],['PROVISIONAL','Provisional']] as [EventFilter,string][]).map(([value,label])=><button key={value} className={eventFilter===value?"active":""} onClick={()=>setEventFilter(value)}>{label}</button>)}</div><label><input type="checkbox" checked={showUnknowns} onChange={e=>setShowUnknowns(e.target.checked)}/> Show unknowns</label></div><div className="tableWrap"><table><thead><tr><th>Plate / Track</th><th>Status</th><th>Confidence</th><th>Camera</th><th>Time</th></tr></thead><tbody>{groupedEvents.map(({main:event,evidence})=><tr className={`eventRow ${event.status.toLowerCase()}`} key={`${event.camera_id}-${event.track_id}`} title={evidence.map(item=>`${item.plate} · ${statusLabel(item.status).toLowerCase()}`).join("\n")}><td className="plate">{event.plate}<small>Track #{event.track_id}{evidence.length>1?` · ${evidence.length} observations`:""}</small></td><td><span className={`badge ${event.status.toLowerCase()}`}>{statusLabel(event.status)}</span></td><td className="confidence">{(event.confidence*100).toFixed(2)}%</td><td>{event.camera_id}</td><td>{mode === "RECORDED_ANALYSIS" ? formatTime(event.video_time_s || 0) : new Date(event.timestamp).toLocaleTimeString()}</td></tr>)}{!groupedEvents.length&&<tr><td colSpan={5} className="none">No meaningful plate events yet</td></tr>}</tbody></table></div></article>
      </aside>
    </section><footer>{mode === "LIVE_ANALYSIS" ? "Every frame is processed sequentially by the live model" : "Preprocessed output from the locked sequential ANPR pipeline"}</footer></>}
  </main>;
}

function Metric({label,value}:{label:string;value:string}){return <div className="metric"><span>{label}</span><strong>{value}</strong></div>}
function formatTime(seconds:number){const safe=Math.max(0,Math.floor(seconds));return `${String(Math.floor(safe/60)).padStart(2,"0")}:${String(safe%60).padStart(2,"0")}`}
function statusLabel(status:ANPREvent["status"]){return status === "ACCEPTED" ? "CONFIRMED" : status === "LIVE_CONSENSUS" ? "UNCONFIRMED" : status.replace("_", " ")}
