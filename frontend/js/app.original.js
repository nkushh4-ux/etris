const pages = ["overview","anpr","tracking","analytics","alerts"];
const titles = {overview:"Overview",anpr:"Live ANPR",tracking:"Vehicle Tracking",analytics:"Traffic Analytics",alerts:"Alert Centre"};

document.querySelectorAll(".nav-item").forEach(btn=>{
  btn.addEventListener("click",()=>showPage(btn.dataset.page));
});
function showPage(page){
  if(!pages.includes(page)) return;
  document.querySelector(".route-panel").classList.remove("expanded");
  document.querySelectorAll(".page").forEach(p=>p.classList.remove("active"));
  document.getElementById(page).classList.add("active");
  document.querySelectorAll(".nav-item").forEach(b=>b.classList.toggle("active",b.dataset.page===page));
  document.getElementById("pageTitle").textContent=titles[page];
  document.querySelectorAll(".nav-item").forEach(b=>{
    if(b.dataset.page===page) b.setAttribute("aria-current","page"); else b.removeAttribute("aria-current");
  });
  if(page==="analytics") buildChart();
}
function updateCamera(){
  const camera=document.getElementById("cameraSelect").value;
  document.getElementById("feedCam").textContent=`${camera} · ${camera==="CAM-04"?"Main Junction":"City Corridor"}`;
  document.querySelector(".anpr-live-heading h3").textContent=`DEMO FEED · ${camera}`;
}
document.getElementById("cameraSelect").addEventListener("change",updateCamera);
function searchVehicle(){
  const q=document.getElementById("vehicleSearch").value.trim().toUpperCase();
  if(!q){ openModal("Enter a registration","Search for the demo vehicle DL 8C AB 1234."); return; }
  if(q.replace(/\s/g, "")==="DL8CAB1234"){
    document.getElementById("vehicleSearch").value="DL 8C AB 1234";
    openModal("Vehicle found", "DL 8C AB 1234 · Sedan · 4 sightings. Last detected at CAM-11, Sector 18, at 10:46 AM.");
    return;
  }
  if(q && q!=="DL 8C AB 1234"){
    openModal("Demo Vehicle Not Found","This prototype contains one predefined demo vehicle: DL 8C AB 1234.");
    document.getElementById("vehicleSearch").value="DL 8C AB 1234";
  }
}
let modalReturnFocus=null;
function openModal(title,text){
  modalReturnFocus=document.activeElement;
  document.getElementById("modalTitle").textContent=title;
  document.getElementById("modalText").textContent=text;
  document.getElementById("modal").classList.add("open");
  document.querySelector(".modal-close").focus();
}
function closeModal(){document.getElementById("modal").classList.remove("open"); modalReturnFocus?.focus();}
function resolveAlert(btn){
  const card=btn.closest(".alert-card");
  card.dataset.type="resolved";
  btn.textContent="Resolved";
  btn.disabled=true;
  const badge=card.querySelector(".badge");
  badge.textContent="Resolved"; badge.className="badge clear";
  applyAlertFilter();
}
const alertsEmpty=document.createElement("p");
alertsEmpty.className="empty-state";
alertsEmpty.setAttribute("role","status");
alertsEmpty.textContent="No incidents in this view.";
document.getElementById("alertsGrid").after(alertsEmpty);
function applyAlertFilter(){
  const filter=document.querySelector(".tab.active").dataset.filter;
  let visible=0;
  document.querySelectorAll(".alert-card").forEach(card=>{
    card.hidden=filter!=="all" && card.dataset.type!==filter;
    if(!card.hidden) visible++;
  });
  alertsEmpty.hidden=visible>0;
}
document.querySelectorAll(".tab").forEach(tab=>{
  tab.setAttribute("aria-pressed",String(tab.classList.contains("active")));
  tab.addEventListener("click",()=>{
    document.querySelectorAll(".tab").forEach(t=>{t.classList.toggle("active",t===tab);t.setAttribute("aria-pressed",String(t===tab));});
    applyAlertFilter();
  });
});
applyAlertFilter();
function buildChart(){
  const chart=document.getElementById("hourChart");
  if(chart.children.length) return;
  [24,31,38,46,52,63,72,78,70,61,49,57,91,82,84,74,68,61].forEach((v,index)=>{
    const i=document.createElement("i"); i.style.height=v+"%"; i.title=`${String(index+6).padStart(2,"0")}:00 · Volume index ${v}/100`; chart.appendChild(i);
  });
}
buildChart();

// Local presentation interactions. No remote data or persistence required.
document.querySelector('.nav-item.active').setAttribute('aria-current','page');
document.getElementById('vehicleSearch').addEventListener('keydown',e=>{if(e.key==='Enter')searchVehicle();});
document.getElementById('modal').addEventListener('click',e=>{if(e.target.id==='modal')closeModal();});
document.addEventListener('keydown',e=>{
  const modal=document.getElementById('modal');
  if(e.key==='Escape'){
    if(modal.classList.contains('open')) closeModal();
    else document.querySelector('.route-panel').classList.remove('expanded');
  }
  if(e.key==='Tab' && modal.classList.contains('open')){
    const controls=modal.querySelectorAll('button');
    const first=controls[0],last=controls[controls.length-1];
    if(e.shiftKey && document.activeElement===first){e.preventDefault();last.focus();}
    else if(!e.shiftKey && document.activeElement===last){e.preventDefault();first.focus();}
  }
});
const detectionEmpty=document.createElement('p');
detectionEmpty.className='empty-state';detectionEmpty.hidden=true;
detectionEmpty.setAttribute('role','status');detectionEmpty.textContent='No matching detections. Try another plate or time.';
document.querySelector('.latest-detections').append(detectionEmpty);
document.getElementById('detectionFilter').addEventListener('input',e=>{
  const query=e.target.value.trim().toLowerCase();let visible=0;
  document.querySelectorAll('.detection-item').forEach(row=>{
    row.hidden=!row.textContent.toLowerCase().includes(query);if(!row.hidden)visible++;
  });detectionEmpty.hidden=visible>0;
});
function exportLog(){
  const rows=[['Timestamp','License plate','Vehicle','Confidence']];
  document.querySelectorAll('.detection-item').forEach(row=>rows.push(Array.from(row.children,cell=>cell.textContent.trim())));
  const csv=rows.map(row=>row.map(cell=>'"'+cell.replace(/"/g,'""')+'"').join(',')).join('\r\n');
  const url=URL.createObjectURL(new Blob(['\ufeff'+csv],{type:'text/csv;charset=utf-8'}));
  const link=document.createElement('a');link.href=url;link.download='ETRIS-demo-detections.csv';document.body.append(link);link.click();link.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
document.querySelectorAll('.camera-marker').forEach(marker=>{
  const id='CAM-'+marker.textContent.slice(1).padStart(2,'0');
  marker.setAttribute('aria-label',`${id}: view demo camera status`);
  marker.addEventListener('click',()=>openModal(id,`${id} · Demo network camera. Traffic status: ${marker.classList.contains('congested')?'Congested':marker.classList.contains('moderate')?'Moderate':'Normal'}.`));
});
const routeMap=document.querySelector('.route-map');
const routeCanvas=document.createElement('div');routeCanvas.className='route-canvas';
Array.from(routeMap.children).filter(node=>!node.classList.contains('map-controls')).forEach(node=>routeCanvas.append(node));
routeMap.prepend(routeCanvas);
let routeZoom=1;
function zoomRoute(direction){routeZoom=Math.max(1,Math.min(1.8,routeZoom+direction*.2));routeCanvas.style.transform=`scale(${routeZoom})`;}
function resetRouteMap(){routeZoom=1;routeCanvas.style.transform='scale(1)';}
const routeTimeline=document.createElement('div');routeTimeline.className='route-timeline';routeTimeline.hidden=true;
routeTimeline.append(document.querySelector('.sightings-panel .timeline').cloneNode(true));routeMap.after(routeTimeline);
function setRouteView(view){
  const isMap=view==='map';routeMap.hidden=!isMap;routeTimeline.hidden=isMap;
  document.querySelector('.route-legend').hidden=!isMap;
  ['map','timeline'].forEach(name=>{const btn=document.getElementById(name+'ViewBtn');btn.classList.toggle('active',name===view);btn.setAttribute('aria-pressed',String(name===view));});
  if(!isMap) document.getElementById('timelineViewBtn').focus();
}
document.getElementById('mapViewBtn').addEventListener('click',()=>setRouteView('map'));
document.getElementById('timelineViewBtn').addEventListener('click',()=>setRouteView('timeline'));
function toggleExpandedMap(){document.querySelector('.route-panel').classList.toggle('expanded');}
setRouteView('map');
