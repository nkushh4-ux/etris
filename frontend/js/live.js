(() => {
  const api = window.ETRIS_API;
  if (!api) return;
  const $ = id => document.getElementById(id);
  const asArray = value => {
    if (Array.isArray(value)) return value;
    if (!value || typeof value !== 'object') return [];
    for (const key of ['items','results','data','events','sightings','records','cameras','trajectory','points','observations']) {
      if (Array.isArray(value[key])) return value[key];
    }
    for (const value of Object.values(value)) if (Array.isArray(value)) return value;
    return [];
  };
  const num = (...values) => { for (const v of values) { const n = Number(v); if (Number.isFinite(n)) return n; } return null; };
  const pick = (obj, keys, fallback = null) => { for (const key of keys) if (obj && obj[key] != null) return obj[key]; return fallback; };
  const text = (el, value, fallback = '—') => { if (el) el.textContent = value ?? fallback; };
  const esc = value => String(value ?? '').replace(/[&<>'"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
  const fmtTime = value => {
    if (value == null || value === '') return '—';
    const d = new Date(value); return Number.isNaN(d.getTime()) ? String(value) : d.toLocaleString([], {hour:'2-digit',minute:'2-digit',second:'2-digit'});
  };
  const normalizePlate = value => String(value ?? '').trim().toUpperCase().replace(/\s+/g,'');

  function eventPlate(e){return pick(e,['plate_text','plate','registration','registration_number','plate_number','text','text_norm'],'—');}
  function eventConfidence(e){const n=num(e?.confidence,e?.plate_confidence,e?.ocr_confidence,e?.score,e?.decision_confidence);return n==null?null:(n<=1?n*100:n);}
  function eventTime(e){return pick(e,['timestamp','time','captured_at','observed_at','created_at','source_time','ts'],'—');}
  function eventVehicle(e){return pick(e,['vehicle_class','class_name','vehicle_type','class','label'],'—');}
  function eventCamera(e){return pick(e,['camera_id','camera','source_camera'],'—');}
  function isVisibleRecognition(e){const plate=String(eventPlate(e)).trim().toUpperCase();const confidence=eventConfidence(e);return plate!==''&&plate!=='UNKNOWN'&&plate!=='—'&&confidence!=null&&confidence>80;}

  async function loadHealth() {
    try { await api.health(); text($('backendStatusText'),'Backend Online'); text($('backendStatusDetail'),api.base); if($('backendPill'))$('backendPill').innerHTML='<span class="dot green"></span> Backend online'; }
    catch { text($('backendStatusText'),'Backend Offline'); text($('backendStatusDetail'),'Start FastAPI on 127.0.0.1:8000'); if($('backendPill'))$('backendPill').innerHTML='<span class="dot red"></span> Backend offline'; }
  }

  let cameraCache=[];
  async function loadCameras(){
    try {
      const raw=await api.tracking.cameras(); const rows=asArray(raw); cameraCache=rows;
      const select=$('cameraSelect');
      if(select){
        const current=select.value; select.innerHTML='<option value="">All cameras</option>';
        rows.forEach((c,i)=>{const id=pick(c,['camera_id','id','name','label'],typeof c==='string'?c:`CAM-${String(i+1).padStart(2,'0')}`);const o=document.createElement('option');o.value=id;o.textContent=id;select.appendChild(o);});
        if([...select.options].some(o=>o.value===current))select.value=current;
      }
      const grid=document.querySelector('.camera-status-grid');
      if(grid){
        if(!rows.length) grid.innerHTML='<p class="empty-state">No cameras returned by backend.</p>';
        else grid.innerHTML=rows.slice(0,8).map((c,i)=>{const id=pick(c,['camera_id','id','name','label'],typeof c==='string'?c:`CAM-${i+1}`);const status=String(pick(c,['status','state'],'AVAILABLE'));const online=!/offline|down|disabled/i.test(status);return `<div><span class="dot ${online?'green':'red'}"></span><b>${esc(id)}</b><small>${esc(status)}</small></div>`;}).join('');
      }
      return rows;
    } catch { return []; }
  }

  function setAnprMedia(mode){
    const stream=$('anprStream'), video=$('anprRecordedVideo'), fallback=$('anprMediaFallback');
    if(!stream||!video||!fallback)return;
    const recorded=/recorded|video|playback/i.test(String(mode||''));
    fallback.hidden=true;
    if(recorded){stream.hidden=true;stream.removeAttribute('src');video.hidden=false;if(!video.src)video.src=api.anpr.recordedVideoUrl();}
    else {video.hidden=true;video.pause();video.removeAttribute('src');stream.hidden=false;if(!stream.src)stream.src=api.anpr.streamUrl();}
  }

  function renderAnprEvents(events){
    const host=$('anprDetectionRows'), live=$('anprLiveRows'); if(!host||!live)return;
    const visible=events.filter(isVisibleRecognition);
    if(!visible.length){host.innerHTML='<p class="empty-state">No recognition above 80% confidence yet.</p>';live.innerHTML='<p class="empty-state">No recognition above 80% confidence yet.</p>';return;}
    host.innerHTML=visible.slice(0,100).map(e=>{const conf=eventConfidence(e);return `<div class="detection-item"><span>${esc(fmtTime(eventTime(e)))}</span><span>${esc(eventPlate(e))}</span><span>${esc(eventVehicle(e))}</span><span class="confidence">${conf.toFixed(1)+'%'}</span></div>`;}).join('');
    live.innerHTML=visible.slice(0,30).map(e=>{const conf=eventConfidence(e);return `<div class="live-item"><span>${esc(fmtTime(eventTime(e)))}</span><span>${esc(eventPlate(e))}</span><span><b>${conf.toFixed(1)+'%'}</b></span></div>`;}).join('');
  }

  async function loadAnpr(){
    text($('anprFeedTitle'),'ANPR FEED · CONNECTING');
    const selectedCamera=$('cameraSelect')?.value || '';
    try {
      const [statusRaw,eventsRaw,recognitionRaw,activeAlertsRaw]=await Promise.all([
        api.anpr.status().catch(()=>({})), api.anpr.events().catch(()=>[]), api.anpr.recognition().catch(()=>null), api.alerts.active().catch(()=>[])
      ]);
      const mode=pick(statusRaw,['mode','analysis_mode','state_mode'],'LIVE_ANALYSIS');
      const state=pick(statusRaw,['status','state','system_state','running'], 'ONLINE');
      const fps=num(statusRaw?.fps,statusRaw?.processing_fps,statusRaw?.current_fps);
      const latency=num(statusRaw?.inference_ms,statusRaw?.latency_ms,statusRaw?.inference_latency_ms,statusRaw?.latency);
      const events=asArray(eventsRaw);
      let filtered=events;
      if(selectedCamera) filtered=events.filter(e=>String(eventCamera(e))===selectedCamera);
      if(!filtered.length && recognitionRaw && typeof recognitionRaw==='object' && !Array.isArray(recognitionRaw)) filtered=[recognitionRaw];
      renderAnprEvents(filtered);
      text($('anprMode'),mode);text($('anprFps'),fps==null?'—':fps.toFixed(1));text($('anprInference'),latency==null?'—':`${latency.toFixed(1)}ms`);
      text($('anprSystemState'),typeof state==='boolean'?(state?'RUNNING':'STOPPED'):String(state).toUpperCase());text($('anprEventCount'),filtered.filter(isVisibleRecognition).length);
      text($('anprActiveCameras'),cameraCache.length||num(statusRaw?.active_cameras,statusRaw?.camera_count)||'—');
      text($('feedCam'),selectedCamera||pick(statusRaw,['camera_id','camera'],'All cameras'));
      text($('anprFeedTitle'),`ANPR FEED · ${String(mode).replaceAll('_',' ')}`);
      setAnprMedia(mode);
      const alerts=asArray(activeAlertsRaw);text($('anprAlertCount'),alerts.length);renderAnprAlerts(alerts,selectedCamera);
    } catch(err){
      text($('anprFeedTitle'),'ANPR FEED · BACKEND UNAVAILABLE');text($('anprSystemState'),'OFFLINE');
      if($('anprDetectionRows'))$('anprDetectionRows').innerHTML=`<p class="empty-state">ANPR API unavailable: ${esc(err.message)}</p>`;
    }
  }

  function renderAnprAlerts(alerts,camera){
    const host=$('anprAlertRows');if(!host)return;const rows=camera?alerts.filter(a=>String(pick(a,['camera_id'],'—'))===camera):alerts;
    if(!rows.length){host.innerHTML='<p class="empty-state">No active alerts returned.</p>';return;}
    host.innerHTML=rows.slice(0,8).map(a=>`<div class="blacklist-alert"><span>!</span><div><b>${esc(String(pick(a,['alert_type','type'],'ALERT')).replaceAll('_',' '))}</b><small>${esc(pick(a,['camera_id'],'—'))} · ${esc(pick(a,['status'],'ACTIVE'))}</small></div></div>`).join('');
  }

  async function anprAction(action){
    try {await api.anpr[action]();if(action==='restart'||action==='play'){const stream=$('anprStream');if(stream){stream.hidden=false;stream.src=api.anpr.streamUrl()+`?t=${Date.now()}`;}}await loadAnpr();}
    catch(err){openModal('ANPR control failed',err.message);}
  }

  function renderAnubhavEvents(raw){
    const host=$('anubhavEvents');if(!host)return;const rows=asArray(raw).filter(e=>{const status=String(pick(e,['status'],'UNKNOWN')).toUpperCase();const confidence=num(e.confidence,e.plate_confidence,e.ocr_confidence);const normalizedConfidence=confidence!=null&&confidence>1?confidence/100:confidence;return status!=='UNKNOWN'&&normalizedConfidence!=null&&normalizedConfidence>.80;});
    if(!rows.length){host.innerHTML='<p class="empty-state">No plate recognition above 80% confidence yet.</p>';return;}
    host.innerHTML=rows.slice(0,12).map(e=>{const plate=pick(e,['plate','plate_text','normalized_text','raw_text'],'UNKNOWN'),status=pick(e,['status'],'PROVISIONAL'),track=pick(e,['track_id'],'—'),rawConfidence=num(e.confidence,e.plate_confidence,e.ocr_confidence),confidence=rawConfidence>1?rawConfidence/100:rawConfidence,time=pick(e,['timestamp','detected_at'],'—');return `<div class="anubhav-event"><div><b>${esc(plate)}</b><small>Track ${esc(track)} · ${esc(fmtTime(time))}</small></div><span>${esc(status)} · ${(confidence*100).toFixed(1)}%</span></div>`;}).join('');
  }

  async function loadAnubhavAnpr(){
    const feed=api.anubhavAnpr;if(!feed){text($('anubhavBackendState'),'RELOAD');const fallback=$('anubhavMediaFallback');if(fallback)fallback.textContent='Frontend assets changed. Refresh this page once.';return;}
    try{
      const [status,events]=await Promise.all([feed.status(),feed.events().catch(()=>[])]);
      text($('anubhavBackendState'),'ONLINE');text($('anubhavPlaybackState'),String(pick(status,['playback_state','status'],status?.running?'PLAYING':'IDLE')).toUpperCase());
      const current=num(status?.current_frame,status?.frame);const total=num(status?.total_frames);text($('anubhavFrame'),current==null?'—':`${current}${total==null?'':` / ${total}`}`);
      const fps=num(status?.inference_fps,status?.fps,status?.processing_fps);text($('anubhavFps'),fps==null?'—':fps.toFixed(1));text($('anubhavFeedCam'),`${pick(status,['camera_id'],'CAM-ANUBHAV')} · port 8001`);
      const fallback=$('anubhavMediaFallback');if(fallback)fallback.hidden=true;renderAnubhavEvents(events);
    }catch(err){text($('anubhavBackendState'),'OFFLINE');text($('anubhavPlaybackState'),'OFFLINE');const fallback=$('anubhavMediaFallback');if(fallback){fallback.hidden=false;fallback.textContent='Anubhav server is offline. Run commands/anubhav_anpr_backend.txt on port 8001.';}}
  }

  async function anubhavAction(action){try{await api.anubhavAnpr[action]();await loadAnubhavAnpr();}catch(err){openModal('Anubhav ANPR control failed',err.message);}}

  async function loadAnalytics() {
    try {
      const [summary,volume,congestion]=await Promise.all([api.analytics.summary().catch(()=>({})),api.analytics.cameraVolume().catch(()=>[]),api.analytics.congestion().catch(()=>[])]);
      const rows=asArray(volume);let total=num(pick(summary,['total_vehicles','vehicle_count','unique_vehicles','total']));
      if(total==null&&rows.length)total=rows.reduce((sum,r)=>sum+(num(r.vehicle_count,r.count,r.volume,r.unique_vehicles)||0),0);
      text($('overviewVehicles'),total!=null?total.toLocaleString():'—');text($('overviewVehiclesNote'),rows.length?`${rows.length} camera-volume records`:'No camera-volume records');text($('analyticsTotalVehicles'),total!=null?total.toLocaleString():'—');
      const congestionRows=asArray(congestion);const severe=congestionRows.filter(r=>/congested|severe/i.test(String(pick(r,['state','status','level','congestion_level'],''))));text($('analyticsCongested'),severe.length||(congestionRows.length?'0':'—'));
      const ratios=congestionRows.map(r=>num(r.ratio,r.congestion_ratio,r.density,r.occupancy)).filter(v=>v!=null);if(ratios.length){const avg=ratios.reduce((a,b)=>a+b,0)/ratios.length;text($('analyticsDensity'),avg<=1?`${Math.round(avg*100)}%`:avg.toFixed(2));}else text($('analyticsDensity'),'—');renderVolumeChart(rows);
    } catch { text($('overviewVehiclesNote'),'Analytics backend unavailable'); }
  }
  function renderVolumeChart(rows){const chart=$('hourChart');if(!chart)return;if(!rows.length){chart.innerHTML='<p class="empty-state">No camera-volume data returned.</p>';return;}chart.innerHTML='';const values=rows.map(r=>num(r.vehicle_count,r.count,r.volume,r.unique_vehicles)||0),max=Math.max(...values,1);rows.slice(0,18).forEach((r,i)=>{const bar=document.createElement('i');bar.style.height=`${Math.max(4,values[i]/max*100)}%`;bar.title=`${pick(r,['camera_id','camera','label','approach_id'],`Record ${i+1}`)} · ${values[i]}`;chart.appendChild(bar);});}

  function alertTypeLabel(a){return pick(a,['alert_type','type'],'ALERT');}function alertStatus(a){return String(pick(a,['status'],'ACTIVE')).toUpperCase();}function alertSeverity(a){return String(pick(a,['severity'],'HIGH')).toUpperCase();}
  async function loadAlerts(){
    const grid=$('alertsGrid');try{const[allRaw,activeRaw]=await Promise.all([api.alerts.list(),api.alerts.active().catch(()=>[])]);const alerts=asArray(allRaw),active=asArray(activeRaw);text($('overviewAlerts'),active.length);text($('overviewAlertsNote'),`${alerts.length} total alert records`);renderOverviewAlerts(active.slice(0,3));if(!grid)return;if(!alerts.length){grid.innerHTML='<p class="empty-state">No alerts returned by backend.</p>';return;}grid.innerHTML=alerts.map(a=>{const sev=alertSeverity(a),status=alertStatus(a),type=alertTypeLabel(a),critical=status!=='CLEARED'&&/critical|high/.test(sev.toLowerCase()),time=pick(a,['detected_at','confirmed_time','entry_time','started_at','created_at','last_updated_at'],'—'),camera=pick(a,['camera_id'],'—'),zone=pick(a,['zone_label','zone_id'],'—'),track=pick(a,['track_id'],'—'),vehicle=pick(a,['vehicle_class'],'—'),blacklist=String(type).toUpperCase()==='BLACKLISTED_VEHICLE',restrictedZone=String(type).toUpperCase()==='RESTRICTED_ZONE',congestionAlert=['CONGESTION','SEVERE_DELAY'].includes(String(type).toUpperCase()),plate=pick(a,['plate_text','plate'],'—'),reason=pick(a,['watchlist_reason'],'—'),confidence=num(pick(a,['plate_confidence','confidence'])),demo=Boolean(pick(a,['demo_watchlist_entry'],false)),inside=num(pick(a,['inside_duration_s'])),authorization=pick(a,['authorization_reason'],'—'),segId=pick(a,['segment_id','zone_id'],'—'),congState=pick(a,['congestion_state'],'—'),delayRatio=num(pick(a,['delay_ratio'])),obsTime=num(pick(a,['observed_travel_time_s'])),basTime=num(pick(a,['baseline_travel_time_s'])),sampleCount=pick(a,['sample_count'],'—'),confirmedAt=pick(a,['confirmed_at'],'—');const evidence=blacklist?`<p><span>Plate</span>${esc(plate)}</p><p><span>Reason</span>${esc(reason)}</p><p><span>Severity</span>${esc(sev)}</p><p><span>Camera</span>${esc(camera)}</p><p><span>Track</span>${esc(track)}</p><p><span>Vehicle</span>${esc(vehicle)}</p><p><span>ANPR Confidence</span>${confidence==null?'—':esc((confidence*100).toFixed(1)+'%')}</p><p><span>Timestamp</span>${esc(time)}</p>${demo?'<p><span>Source</span>DEMO WATCHLIST ENTRY</p>':''}`:restrictedZone?`<p><span>Zone</span>${esc(zone)}</p><p><span>Vehicle</span>${esc(vehicle)}</p><p><span>Track</span>${esc(track)}</p><p><span>Accepted Plate</span>${esc(plate)}</p><p><span>Camera</span>${esc(camera)}</p><p><span>Entry Time</span>${esc(pick(a,['entry_time'],'—'))}</p><p><span>Inside Duration</span>${inside==null?'—':esc(inside.toFixed(1)+' s')}</p><p><span>Authorization</span>${esc(authorization)}</p>`:congestionAlert?`<p><span>Segment</span>${esc(segId)}</p><p><span>State</span>${esc(congState)}</p><p><span>Delay Ratio</span>${delayRatio==null?'—':esc(delayRatio.toFixed(2)+'×')}</p><p><span>Observed Travel</span>${obsTime==null?'—':esc(obsTime.toFixed(1)+' s')}</p><p><span>Baseline Travel</span>${basTime==null?'—':esc(basTime.toFixed(1)+' s')}</p><p><span>Samples</span>${esc(String(sampleCount))}</p><p><span>Confirmed At</span>${esc(String(confirmedAt))}</p><p><span>Status</span>${esc(status)}</p>`:`<p><span>Vehicle</span>${esc(vehicle)}</p><p><span>Track</span>${esc(track)}</p><p><span>Zone</span>${esc(zone)}</p>`;return `<article class="panel alert-card" data-type="${status==='CLEARED'?'resolved':critical?'critical':'warning'}"><div class="alert-card-head"><span class="big-alert ${critical?'critical':'warning'}">${critical?'!':'•'}</span><div><small>${esc(time)} · ${esc(camera)}</small><h2>${esc(type.replaceAll('_',' '))}</h2></div><span class="badge ${status==='CLEARED'?'clear':critical?'danger':'warning'}">${esc(status)}</span></div><div class="alert-details">${evidence}</div></article>`;}).join('');if(typeof applyAlertFilter==='function')applyAlertFilter();}catch(err){if(grid)grid.innerHTML=`<p class="empty-state">Alert backend unavailable: ${esc(err.message)}</p>`;text($('overviewAlerts'),'—');text($('overviewAlertsNote'),'Alert API unavailable');renderOverviewAlerts([]);}}
  function renderOverviewAlerts(items){const host=$('overviewAlertList');if(!host)return;if(!items.length){host.innerHTML='<p class="empty-state">No active alerts returned.</p>';return;}host.innerHTML=items.map(a=>`<div class="mini-alert warning"><span class="alert-icon">!</span><div><b>${esc(alertTypeLabel(a).replaceAll('_',' '))}</b><small>${esc(pick(a,['camera_id'],'—'))} · ${esc(pick(a,['status'],'ACTIVE'))}</small></div></div>`).join('');}

  async function loadSignal(){const host=$('signalApproaches');try{const raw=await api.signal.demo(),plan=raw?.plan||raw?.baseline||raw,approaches=asArray(plan?.approaches||plan?.recommendations||plan?.phases||raw?.approaches||raw?.recommendations),cycle=num(plan?.cycle_duration_s,plan?.cycle_s,plan?.cycle_duration,raw?.cycle_duration_s);text($('signalCycle'),cycle!=null?`${cycle}s`:'—');text($('signalState'),'Online');text($('signalStateNote'),'Recommendation loaded');let priority='—';if(approaches.length){const sorted=[...approaches].sort((a,b)=>(num(b.priority,b.pressure,b.pressure_score)||0)-(num(a.priority,a.pressure,a.pressure_score)||0));priority=pick(sorted[0],['approach_id','id','name','label'],'—');}text($('signalPriority'),priority);if(!host)return;if(!approaches.length){host.innerHTML=`<pre class="api-json">${esc(JSON.stringify(raw,null,2))}</pre>`;return;}host.innerHTML=approaches.map(a=>{const id=pick(a,['approach_id','id','name','label'],'Approach'),pressure=num(a.pressure,a.pressure_score,a.priority),green=num(a.green_s,a.green_seconds,a.recommended_green_s,a.green_duration_s,a.green_duration),reason=pick(a,['reason','reason_code','reason_codes'],'Traffic demand');return `<div class="signal-row"><div><b>${esc(id)}</b><small>${esc(Array.isArray(reason)?reason.join(' · '):reason)}</small></div><div><span>Pressure</span><strong>${pressure!=null?pressure.toFixed(3):'—'}</strong></div><div><span>Green</span><strong>${green!=null?green+'s':'—'}</strong></div></div>`;}).join('');}catch(err){text($('signalState'),'Offline');text($('signalStateNote'),err.message);if(host)host.innerHTML=`<p class="empty-state">Signal-control API unavailable: ${esc(err.message)}</p>`;}}

  function extractSightings(raw){const rows=asArray(raw);if(rows.length)return rows;if(raw&&typeof raw==='object'){const nested=pick(raw,['sightings','events','observations','points']);if(Array.isArray(nested))return nested;}return [];}
  function extractTrajectory(raw){const rows=asArray(raw);if(rows.length)return rows;if(raw&&typeof raw==='object'){for(const key of ['sightings','trajectory','points','segments','events'])if(Array.isArray(raw[key]))return raw[key];}return [];}
  function sightingTime(s){return pick(s,['timestamp','observed_at','captured_at','time','source_time','ts'],'—');}
  function sightingCamera(s){return pick(s,['camera_id','camera','source_camera','id'],'—');}

  async function searchVehicle(){
    const input=$('vehicleSearch');const rawPlate=input?.value.trim()||'';if(!rawPlate){openModal('Enter a registration','Enter a vehicle registration number to query the sightings and trajectory APIs.');return;}
    const plate=normalizePlate(rawPlate);text($('trackingPlate'),plate);text($('trackingStatus'),'Searching…');
    try{
      const[sRaw,tRaw]=await Promise.all([api.tracking.byPlate(plate),api.tracking.trajectory(plate).catch(()=>({}))]);
      let sightings=extractSightings(sRaw);let trajectory=extractTrajectory(tRaw);if(!sightings.length&&trajectory.length)sightings=trajectory;
      sightings=[...sightings].sort((a,b)=>new Date(sightingTime(a))-new Date(sightingTime(b)));
      renderTracking(plate,sightings,tRaw,trajectory);
      if(!sightings.length)openModal('No sightings',`The backend returned no sightings for ${plate}.`);
    }catch(err){renderTracking(plate,[],{},[]);text($('trackingStatus'),'Unavailable');openModal('Vehicle lookup failed',err.message);}
  }
  window.ETRIS_searchVehicle=searchVehicle;

  function renderTracking(plate,sightings,tRaw,trajectory){
    text($('trackingPlate'),plate);const first=sightings[0],last=sightings[sightings.length-1];const vehicle=pick(last||first,['vehicle_class','vehicle_type','class_name','class'],'—');
    text($('trackingVehicleType'),vehicle);text($('trackingVehicleSymbol'),vehicle==='—'?'VEHICLE':String(vehicle).toUpperCase());text($('trackingFirst'),first?fmtTime(sightingTime(first)):'—');text($('trackingLast'),last?fmtTime(sightingTime(last)):'—');
    text($('trackingSightingsCount'),sightings.length);text($('trackingTotalSightings'),sightings.length);const cams=new Set(sightings.map(sightingCamera).filter(v=>v&&v!=='—'));text($('trackingCameraCount'),`${cams.size} camera${cams.size===1?'':'s'}`);text($('trackingStatus'),sightings.length?'Found':'No sightings');
    const distance=num(tRaw?.total_distance_km,tRaw?.distance_km,tRaw?.total_distance_m,tRaw?.distance_m);let distanceLabel='—';if(distance!=null){const isM=tRaw?.total_distance_m!=null||tRaw?.distance_m!=null;distanceLabel=isM?`${(distance/1000).toFixed(2)} km`:`${distance.toFixed(2)} km`;}text($('trackingDistance'),distanceLabel);
    const speed=num(tRaw?.average_speed_kmh,tRaw?.avg_speed_kmh,tRaw?.speed_kmh);text($('trackingSpeed'),speed==null?'—':`${speed.toFixed(1)} km/h`);
    let duration=num(tRaw?.duration_s,tRaw?.travel_time_s,tRaw?.total_time_s);if(duration==null&&first&&last){const a=new Date(sightingTime(first)),b=new Date(sightingTime(last));if(!Number.isNaN(a.getTime())&&!Number.isNaN(b.getTime()))duration=(b-a)/1000;}text($('trackingDuration'),duration==null?'—':duration<60?`${Math.round(duration)}s`:`${Math.round(duration/60)}m`);
    renderTrackingTimeline(sightings);renderTrackingRoute(trajectory.length?trajectory:sightings);
  }
  function renderTrackingTimeline(sightings){const host=$('trackingTimeline');if(!host)return;if(!sightings.length){host.innerHTML='<p class="empty-state">No sightings returned for this vehicle.</p>';return;}host.innerHTML=[...sightings].reverse().map((s,i)=>`<div class="timeline-item ${i===0?'last':''}"><i></i><div class="timeline-content"><div class="timeline-title"><b>${esc(sightingCamera(s))}</b><span class="timeline-time">${esc(fmtTime(sightingTime(s)))}</span>${i===0?'<em>Latest</em>':''}</div><span class="timeline-location">${esc(pick(s,['location','camera_name','zone','direction'],'Backend sighting'))}</span></div></div>`).join('');}
  function renderTrackingRoute(points){const locations=$('trackingRouteLocations'),svg=$('trackingRouteSvg'),empty=$('trackingRouteEmpty');if(!locations||!svg)return;locations.innerHTML='';svg.innerHTML='';if(empty)empty.hidden=points.length>0;if(!points.length)return;const count=points.length;const coords=points.map((p,i)=>{const x=count===1?50:14+(72*i/(count-1));const wave=[35,65,28,70,42,62];const y=wave[i%wave.length];return{x,y,p};});coords.forEach(({x,y,p},i)=>{if(i){const prev=coords[i-1];svg.insertAdjacentHTML('beforeend',`<line class="route-line route-line-cyan" x1="${prev.x}" y1="${prev.y}" x2="${x}" y2="${y}"/>`);}const node=document.createElement('div');node.className='route-location';node.style.left=`${x}%`;node.style.top=`${y}%`;node.style.transform='translate(-50%,-50%)';node.innerHTML=`<div class="location-label"><b>${esc(sightingCamera(p))}</b><span>${esc(fmtTime(sightingTime(p)))}</span></div><div class="location-dot ${i===count-1?'current':'detected'}"></div>`;locations.appendChild(node);});}

  document.addEventListener('DOMContentLoaded',async()=>{
    loadHealth();await loadCameras();loadAnalytics();loadAlerts();loadSignal();loadAnpr();
    $('refreshSignalBtn')?.addEventListener('click',loadSignal);$('anprRefreshBtn')?.addEventListener('click',loadAnpr);$('cameraSelect')?.addEventListener('change',loadAnpr);
    $('anprPlayBtn')?.addEventListener('click',()=>anprAction('play'));$('anprPauseBtn')?.addEventListener('click',()=>anprAction('pause'));$('anprRestartBtn')?.addEventListener('click',()=>anprAction('restart'));
    $('anprRate')?.addEventListener('change',async e=>{try{await api.anpr.processingRate(e.target.value);await loadAnpr();}catch(err){openModal('Processing rate failed',err.message);}});
    $('anprStream')?.addEventListener('error',()=>{const stream=$('anprStream'),fallback=$('anprMediaFallback');if(fallback){fallback.hidden=false;fallback.textContent='Live stream reconnecting…';}setTimeout(()=>{if(stream){stream.hidden=false;stream.src=api.anpr.streamUrl()+`?t=${Date.now()}`;}},1800);});
    const anubhavStream=$('anubhavStream');if(anubhavStream&&api.anubhavAnpr){anubhavStream.src=api.anubhavAnpr.streamUrl()+`?t=${Date.now()}`;anubhavStream.addEventListener('error',()=>{const fallback=$('anubhavMediaFallback');if(fallback){fallback.hidden=false;fallback.textContent='Stream reconnecting…';}setTimeout(()=>{anubhavStream.src=api.anubhavAnpr.streamUrl()+`?t=${Date.now()}`;},1800);});}
    $('anubhavPlayBtn')?.addEventListener('click',()=>anubhavAction('play'));$('anubhavPauseBtn')?.addEventListener('click',()=>anubhavAction('pause'));$('anubhavRestartBtn')?.addEventListener('click',()=>anubhavAction('restart'));loadAnubhavAnpr();
    setInterval(loadHealth,15000);setInterval(loadAnpr,4000);setInterval(loadAnubhavAnpr,4000);
  });
})();
