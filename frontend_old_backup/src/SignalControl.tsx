import {useCallback, useEffect, useMemo, useRef, useState} from "react";
import type {RealTrafficEvent, SignalCyclePlan, SignalDecision} from "./services/signalControlService";
import {getBaselineSignalPlan, getEastSurgeSignalPlan, getRealTrafficEvents, getTrafficDemoVideoURL} from "./services/signalControlService";

type Scenario = "BASELINE" | "EAST_DEMAND_SURGE";

export default function SignalControl() {
  const [source,setSource]=useState<"REAL"|"SYNTHETIC">("REAL");
  return <><div className="signalSourceTabs"><button className={source==="REAL"?"active":""} onClick={()=>setSource("REAL")}>REAL VIDEO ESTIMATE</button><button className={source==="SYNTHETIC"?"active":""} onClick={()=>setSource("SYNTHETIC")}>SYNTHETIC SCENARIOS</button></div>{source==="REAL"?<RealTrafficControl/>:<SyntheticSignalControl/>}</>;
}

function SyntheticSignalControl() {
  const [scenario, setScenario] = useState<Scenario>("BASELINE");
  const [plan, setPlan] = useState<SignalCyclePlan | null>(null);
  const [baseline, setBaseline] = useState<SignalCyclePlan | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const load = useCallback(async (next: Scenario) => {
    setLoading(true); setError("");
    try {
      const response = next === "BASELINE" ? await getBaselineSignalPlan() : await getEastSurgeSignalPlan();
      setPlan(response.plan); setScenario(next);
      if (next === "BASELINE") setBaseline(response.plan);
      else if (!baseline) {
        const original = await getBaselineSignalPlan(); setBaseline(original.plan);
      }
    } catch (reason) {
      setError(reason instanceof TypeError || (reason instanceof DOMException && reason.name === "TimeoutError")
        ? "Signal-control backend is offline." : reason instanceof Error ? reason.message : "Signal plan request failed.");
    } finally { setLoading(false); }
  }, [baseline]);

  useEffect(() => { void load("BASELINE"); }, []);
  const baselineByApproach = useMemo(() => new Map(baseline?.ordered_phases.map(item=>[item.approach_id,item.green_duration_s]) || []), [baseline]);
  const eastCurrent = plan?.ordered_phases.find(item=>item.approach_id==="EAST");
  const eastBaseline = baselineByApproach.get("EAST");

  return <section className="signalView">
    <div className="signalHero"><div><span className="eyebrow">ADAPTIVE SIGNAL CONTROL</span><h2>Recommendation Mode</h2><p>Deterministic phase planning from current approach demand.</p></div><div className="simulationNotice"><strong>SIMULATION / RECOMMENDATION ONLY</strong><span>No physical traffic-signal hardware is being controlled.</span></div></div>
    <div className="scenarioBar"><div><button className={scenario==="BASELINE"?"active":""} disabled={loading} onClick={()=>void load("BASELINE")}>BASELINE TRAFFIC</button><button className={scenario==="EAST_DEMAND_SURGE"?"active surge":""} disabled={loading} onClick={()=>void load("EAST_DEMAND_SURGE")}>SIMULATE EAST SURGE</button></div>{scenario==="EAST_DEMAND_SURGE"&&eastCurrent&&eastBaseline!==undefined&&<div className="surgeSummary"><strong>EAST demand increased</strong><span>{eastBaseline}s → {eastCurrent.green_duration_s}s green</span></div>}</div>
    {loading && !plan && <SignalState title="Loading signal plan…" detail="Requesting the deterministic synthetic demonstration."/>}
    {error && <SignalState title={error} detail="No signal recommendation is available until the API responds." action={()=>void load(scenario)}/>} 
    {!loading && !error && (!plan || !plan.ordered_phases.length) && <SignalState title="No approaches in this plan." detail="Submit approach traffic state to generate a recommendation." action={()=>void load("BASELINE")}/>} 
    {plan && !error && <>
      {loading && <div className="planUpdating">Updating recommendation…</div>}
      <div className="signalMeta"><span>INTERSECTION <strong>{plan.intersection_id}</strong></span><span>PHASES <strong>{plan.ordered_phases.length}</strong></span><span>CYCLE <strong>{plan.cycle_duration_s}s</strong></span><span>GENERATED <strong>{new Date(plan.generated_at).toLocaleTimeString()}</strong></span></div>
      <div className="approachCards">{plan.ordered_phases.map(decision=><ApproachCard key={decision.approach_id} decision={decision} baselineGreen={baselineByApproach.get(decision.approach_id)} showDelta={scenario!=="BASELINE"}/>)}</div>
      <div className="signalLower">
        <article className="panel phasePanel"><div className="panelTitle"><span>PHASE ORDER</span><small>Total cycle · {plan.cycle_duration_s}s</small></div><ol>{plan.ordered_phases.map((phase,index)=><li key={phase.approach_id}><span>{String(index+1).padStart(2,"0")}</span><strong>{phase.approach_id}</strong><b>{phase.green_duration_s}s</b></li>)}</ol><div className="cycleFoot">Includes configured yellow and all-red transition overhead</div></article>
        <article className="panel pressurePanel"><div className="panelTitle"><span>TRAFFIC PRESSURE</span><small>Normalized · 0–1</small></div><div>{plan.ordered_phases.map(phase=><div className="pressureRow" key={phase.approach_id}><span>{phase.approach_id}</span><div><i style={{width:`${Math.max(0,Math.min(1,phase.pressure))*100}%`}}/></div><strong>{phase.pressure.toFixed(3)}</strong></div>)}</div></article>
      </div>
      <div className="recommendationFoot">Plan output is advisory and intended for simulation or future supervised controller integration.</div>
    </>}
  </section>;
}

function RealTrafficControl() {
  const [events,setEvents]=useState<RealTrafficEvent[]>([]); const [videoTime,setVideoTime]=useState(0);
  const [videoDuration,setVideoDuration]=useState(0);
  const [loading,setLoading]=useState(true); const [error,setError]=useState(""); const videoRef=useRef<HTMLVideoElement>(null);
  const load=useCallback(async()=>{setLoading(true);setError("");try{setEvents(await getRealTrafficEvents())}catch(reason){setError(reason instanceof TypeError?"Traffic demo backend is offline.":reason instanceof Error?reason.message:"Traffic demo unavailable.")}finally{setLoading(false)}},[]);
  useEffect(()=>{void load()},[load]);
  const current=useMemo(()=>{let found:RealTrafficEvent|undefined;for(const event of events){if(event.video_time_s<=videoTime)found=event;else break}return found||events[0]},[events,videoTime]);
  const decisions=useMemo(()=>new Map(current?.approaches.map(item=>[item.approach_id,item.controller])||[]),[current]);
  return <section className="signalView realSignalView">
    <div className="signalHero"><div><span className="eyebrow">ADAPTIVE SIGNAL CONTROL</span><h2>Real Video Estimate</h2><p>Preprocessed vehicle detections synchronized with source-video time.</p></div><div className="simulationNotice"><strong>RECOMMENDATION ONLY</strong><span>No physical traffic-signal hardware is being controlled.</span></div></div>
    {loading&&<SignalState title="Loading real traffic-state events…" detail="Reading preprocessed model output."/>}
    {error&&<SignalState title={error} detail="Generate the full traffic demo output and retry." action={()=>void load()}/>} 
    {!loading&&!error&&!current&&<SignalState title="No traffic-state events available." detail="The generated JSON contains no controller updates." action={()=>void load()}/>} 
    {current&&!error&&<div className="realSignalLayout"><article className="panel realVideoPanel"><div className="panelTitle"><span>PREPROCESSED REAL MODEL OUTPUT</span><small>Frame {current.frame_index} · {current.video_time_s.toFixed(1)}s</small></div><video ref={videoRef} src={getTrafficDemoVideoURL()} controls preload="metadata" onLoadedMetadata={event=>setVideoDuration(event.currentTarget.duration)} onTimeUpdate={event=>setVideoTime(event.currentTarget.currentTime)} onSeeked={event=>setVideoTime(event.currentTarget.currentTime)}/><div className="syncLine"><i style={{width:`${videoDuration>0?Math.min(100,videoTime/videoDuration*100):0}%`}}/><span>Controller state follows video time · latest update ≤ {videoTime.toFixed(1)}s</span></div></article><div className="realSignalSide"><div className="realApproaches">{current.approaches.map(item=><article className="approachCard" key={item.approach_id}><div className="approachHead"><div><strong>{item.label}</strong><span>{item.approach_id}</span></div>{item.controller.fairness_applied&&<span className="fairnessBadge">STARVATION GUARD</span>}</div><div className="greenTime"><strong>{item.controller.green_duration_s}</strong><span>s GREEN</span></div><div className="realMetrics"><span>Vehicles<b>{item.vehicle_count}</b></span><span>Occupancy<b>{(item.occupancy*100).toFixed(1)}%</b></span><span>Arrivals<b>{item.arrival_rate_vpm.toFixed(1)}/min</b></span><span>Queue estimate<b>{item.queue_length}</b></span><span>Average wait<b>{item.average_waiting_time_s.toFixed(1)}s</b></span><span>Pressure<b>{item.controller.pressure.toFixed(3)}</b></span></div><div className="pressureInline"><i style={{width:`${item.controller.pressure*100}%`}}/></div><div className="reasonList">{item.controller.reason_codes.map(reason=><span key={reason}>{reasonLabel(reason)}</span>)}</div></article>)}</div><article className="panel compactPhase"><div className="panelTitle"><span>CURRENT PHASE ORDER</span><small>Cycle · {current.cycle_duration_s}s</small></div><ol>{current.phase_order.map((id,index)=><li key={id}><span>{index+1}</span><strong>{id}</strong><b>{decisions.get(id)?.green_duration_s ?? "—"}s</b></li>)}</ol></article></div></div>}
    <div className="estimateLimits"><span>Occupancy = image-space estimate</span><span>Queue = slow/stationary congestion estimate</span><span>Signal plan = recommendation only</span></div>
  </section>;
}

function ApproachCard({decision,baselineGreen,showDelta}:{decision:SignalDecision;baselineGreen?:number;showDelta:boolean}) {
  const delta = baselineGreen === undefined ? null : decision.green_duration_s-baselineGreen;
  return <article className={`approachCard approach-${decision.approach_id.toLowerCase()}`}><div className="approachHead"><strong>{decision.approach_id}</strong>{decision.fairness_applied&&<span className="fairnessBadge">STARVATION GUARD</span>}</div><div className="greenTime"><strong>{decision.green_duration_s}</strong><span>s GREEN</span>{showDelta&&delta!==null&&<b className={delta>0?"up":delta<0?"down":""}>{delta>0?"+":""}{delta}s</b>}</div><dl><div><dt>Pressure</dt><dd>{decision.pressure.toFixed(3)}</dd></div><div><dt>Priority</dt><dd>{priorityLabel(decision.effective_priority)} · {decision.effective_priority.toFixed(3)}</dd></div></dl><div className="reasonList">{decision.reason_codes.map(reason=><span key={reason}>{reasonLabel(reason)}</span>)}</div></article>;
}
function SignalState({title,detail,action}:{title:string;detail:string;action?:()=>void}){return <div className="signalState"><strong>{title}</strong><span>{detail}</span>{action&&<button onClick={action}>Retry</button>}</div>}
function priorityLabel(value:number){return value>=.75?"HIGH":value>=.5?"ELEVATED":value>=.25?"MEDIUM":"LOW"}
function reasonLabel(value:string){return value==="FAIRNESS_STARVATION_GUARD"?"FAIRNESS BOOST":value.replaceAll("_"," ")}
