(() => {
  const defaultBase = 'http://127.0.0.1:8000';
  const base = (window.ETRIS_API_BASE_URL || localStorage.getItem('etrisApiBase') || defaultBase).replace(/\/$/, '');
  const anubhavBase = (window.ETRIS_ANUBHAV_API_BASE_URL || localStorage.getItem('etrisAnubhavApiBase') || 'http://127.0.0.1:8001').replace(/\/$/, '');

  const query = params => {
    const q = new URLSearchParams(Object.entries(params || {}).filter(([,v]) => v != null && v !== ''));
    return q.toString() ? `?${q}` : '';
  };

  async function request(path, options = {}) {
    return requestFrom(base, path, options);
  }

  async function requestFrom(apiBase, path, options = {}) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), options.timeoutMs || 8000);
    try {
      const response = await fetch(apiBase + path, {
        ...options,
        signal: controller.signal,
        headers: { 'Accept': 'application/json', ...(options.headers || {}) }
      });
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      const type = response.headers.get('content-type') || '';
      return type.includes('application/json') ? response.json() : response.text();
    } finally { clearTimeout(timeout); }
  }

  window.ETRIS_API = {
    base,
    health: () => request('/openapi.json', { timeoutMs: 3500 }),
    url: path => base + path,
    anpr: {
      status: (mode = 'LIVE_ANALYSIS') => request('/api/anpr/status' + query({ mode })),
      events: () => request('/api/anpr/events'),
      recognition: () => request('/api/anpr/recognition'),
      streamUrl: () => base + '/api/anpr/stream',
      recordedVideoUrl: () => base + '/api/anpr/recorded-video',
      recordedEvents: () => request('/api/anpr/recorded-events'),
      play: () => request('/api/anpr/play', { method: 'POST' }),
      pause: () => request('/api/anpr/pause', { method: 'POST' }),
      restart: () => request('/api/anpr/restart', { method: 'POST' }),
      processingRate: rate => request('/api/anpr/processing-rate', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ rate })
      })
    },
    anubhavAnpr: {
      base: anubhavBase,
      status: () => requestFrom(anubhavBase, '/api/anpr/status' + query({ mode: 'LIVE_ANALYSIS' })),
      events: () => requestFrom(anubhavBase, '/api/anpr/events'),
      streamUrl: () => anubhavBase + '/api/anpr/stream',
      play: () => requestFrom(anubhavBase, '/api/anpr/play', { method: 'POST' }),
      pause: () => requestFrom(anubhavBase, '/api/anpr/pause', { method: 'POST' }),
      restart: () => requestFrom(anubhavBase, '/api/anpr/restart', { method: 'POST' })
    },
    tracking: {
      recent: (limit = 50, camera_id = null) => request('/api/sightings/recent' + query({ limit, camera_id })),
      byPlate: (plate, start_time = null, end_time = null) => request(`/api/sightings/${encodeURIComponent(plate)}` + query({ start_time, end_time })),
      cameras: () => request('/api/cameras'),
      trajectory: (plate, start_time = null, end_time = null) => request(`/api/trajectory/${encodeURIComponent(plate)}` + query({ start_time, end_time }))
    },
    analytics: {
      summary: () => request('/api/analytics/summary'),
      cameraVolume: () => request('/api/analytics/camera-volume'),
      od: () => request('/api/analytics/od'),
      segments: () => request('/api/analytics/segments'),
      congestion: () => request('/api/analytics/congestion'),
      heatmap: () => request('/api/analytics/heatmap'),
      routes: () => request('/api/analytics/routes')
    },
    signal: {
      demo: () => request('/api/signal-control/demo'),
      demoChange: () => request('/api/signal-control/demo/change'),
      plan: payload => request('/api/signal-control/plan', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
      }),
      trafficDemoEvents: () => request('/api/signal-control/traffic-demo/events'),
      trafficDemoVideoUrl: () => base + '/api/signal-control/traffic-demo/video'
    },
    alerts: {
      list: (params = {}) => request('/api/alerts' + query(params)),
      active: (params = {}) => request('/api/alerts/active' + query(params)),
      get: id => request(`/api/alerts/${encodeURIComponent(id)}`)
    },
    watchlist: {
      list: () => request('/api/watchlist'),
      add: payload => request('/api/watchlist', {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
      }),
      update: (id, payload) => request(`/api/watchlist/${encodeURIComponent(id)}`, {
        method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload)
      }),
      deactivate: id => request(`/api/watchlist/${encodeURIComponent(id)}`, { method: 'DELETE' })
    }
  };
})();
