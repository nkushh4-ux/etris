# ETRIS Command Centre Frontend

ETRIS — Enhanced Traffic Recognition & Intelligence System

This package preserves the Janhavi UI/UX design and connects it to the existing ETRIS FastAPI backend. It does not contain the AI models or database itself; it consumes the backend API running separately.

## Pages
- Overview
- Live ANPR
- Vehicle Tracking
- Traffic Analytics
- Signal Control
- Alert Centre

## Default API
`http://127.0.0.1:8000`

Backend health is checked through `/openapi.json`.

### ANPR
- `GET /api/anpr/status?mode=LIVE_ANALYSIS`
- `GET /api/anpr/events`
- `GET /api/anpr/recognition`
- `GET /api/anpr/stream`
- `GET /api/anpr/recorded-video`
- `GET /api/anpr/recorded-events`
- `POST /api/anpr/play`
- `POST /api/anpr/pause`
- `POST /api/anpr/restart`
- `POST /api/anpr/processing-rate` with `{ "rate": "1x" }`

### Tracking / Stage 9
- `GET /api/sightings/recent`
- `GET /api/sightings/{plate_text}`
- `GET /api/cameras`
- `GET /api/trajectory/{plate_text}`

### Analytics
- `GET /api/analytics/summary`
- `GET /api/analytics/camera-volume`
- `GET /api/analytics/od`
- `GET /api/analytics/segments`
- `GET /api/analytics/congestion`
- `GET /api/analytics/heatmap`
- `GET /api/analytics/routes`

### Signal Control
- `GET /api/signal-control/demo`
- `GET /api/signal-control/demo/change`
- `POST /api/signal-control/plan`
- `GET /api/signal-control/traffic-demo/events`
- `GET /api/signal-control/traffic-demo/video`

Signal Control remains recommendation/simulation only; the frontend does not claim physical signal actuation.

### Alerts
- `GET /api/alerts`
- `GET /api/alerts/active`
- `GET /api/alerts/{alert_id}`

## Run frontend
From this `ETRIS` folder:

```powershell
python -m http.server 5173
```

Open:
`http://127.0.0.1:5173`

## Run backend
From `B:\ETRIS ultimate AI`:

```powershell
$env:PYTHONPATH=(Get-Location).Path
$env:DATABASE_URL="postgresql+psycopg://etris:etris@127.0.0.1:5432/etris"
$env:YOLO_CONFIG_DIR=(Resolve-Path ".ultralytics-config").Path

.\.venv-ocr-svtrv2\Scripts\python.exe -m uvicorn backend.api.app:app --host 127.0.0.1 --port 8000
```

## CORS
Because the frontend and backend run on different ports, FastAPI must allow:
- `http://127.0.0.1:5173`
- `http://localhost:5173`

## API response handling
Several OpenAPI response schemas are intentionally untyped (`schema: {}`). The frontend therefore uses defensive adapters that recognize common backend field names. Missing fields are displayed as `—`; no plate, vehicle, trajectory, latency, count, or alert value is fabricated.

## Override API base URL
Before `js/api.js` loads, set `window.ETRIS_API_BASE_URL`, or run:

```js
localStorage.setItem('etrisApiBase','http://127.0.0.1:8000')
```
