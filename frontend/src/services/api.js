const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');
const POLL_INTERVAL_MS = 700;
const POLL_TIMEOUT_MS = 60 * 60 * 1000;
const DOWNLOAD_TIMEOUT_MS = 30 * 60 * 1000;
const DEFAULT_FRAME_INTERVAL = import.meta.env.VITE_DEFAULT_FRAME_INTERVAL || '1.0';
const DEFAULT_CONFIDENCE_THRESHOLD = import.meta.env.VITE_DEFAULT_CONFIDENCE_THRESHOLD || '0.3';

const apiUrl = (path) => `${API_BASE_URL}${path}`;

const errorMessage = (payload) => {
  const detail = payload?.error?.message || payload?.detail;
  if (Array.isArray(detail)) return detail.map((item) => item.msg || String(item)).join('; ');
  return detail || 'HydroWatch request failed.';
};

const requestJson = async (path, options = {}) => {
  let response;
  try {
    response = await fetch(apiUrl(path), options);
  } catch (error) {
    if (error.name === 'AbortError') throw error;
    throw new Error('Cannot reach the HydroWatch backend. Start it with "python run.py" in the backend folder.');
  }
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(errorMessage(payload));
  return payload;
};

const densityLabel = (density) => {
  if (!density) return '0 debris per frame';
  if (typeof density === 'string') return density;
  return `${density.value} ${density.unit}`;
};

const adaptFrame = (frame) => ({
  ...frame,
  mediaUrl: frame.preview_url,
  density: densityLabel(frame.density),
  detections: (frame.detections || []).map((detection) => ({
    ...detection,
    confidence: Number((detection.confidence * 100).toFixed(1)),
    x: detection.bbox.x * 100,
    y: detection.bbox.y * 100,
    width: detection.bbox.width * 100,
    height: detection.bbox.height * 100,
  })),
});

const adaptAnalysis = (analysis) => ({
  ...analysis,
  totalDetections: analysis.total_detections,
  debrisDetections: analysis.debris_detections,
  averageConfidence: analysis.average_confidence,
  sizeStats: analysis.size_stats,
  density: densityLabel(analysis.density),
  peakDebris: analysis.peak_debris_per_frame,
  frames: (analysis.frames || []).map(adaptFrame),
  location: analysis.location?.label || analysis.location?.query || '',
  locationInfo: analysis.location || null,
});

const pollAnalysis = async (analysisId, onProgress, signal) => {
  const startedAt = Date.now();
  while (Date.now() - startedAt < POLL_TIMEOUT_MS) {
    if (signal?.aborted) throw new DOMException('Analysis polling cancelled.', 'AbortError');
    const analysis = await requestJson(`/api/v1/analyses/${analysisId}`, { signal });
    onProgress?.(analysis);
    if (analysis.status === 'completed') return adaptAnalysis(analysis);
    if (analysis.status === 'failed') throw new Error(analysis.message || 'Analysis failed.');
    await new Promise((resolve, reject) => {
      const timer = setTimeout(resolve, POLL_INTERVAL_MS);
      signal?.addEventListener('abort', () => { clearTimeout(timer); reject(new DOMException('Analysis polling cancelled.', 'AbortError')); }, { once: true });
    });
  }
  throw new Error('Analysis timed out while processing the submitted media.');
};

// location: a place-name string, or { name, latitude, longitude, waterBody, depth, capturedAt, notes }
export const analyzeVideo = async (files, location, onProgress, signal) => {
  const body = new FormData();
  files.forEach((file) => body.append('files', file));
  const loc = typeof location === 'string' ? { name: location } : (location || {});
  const fields = {
    location: loc.name,
    latitude: loc.latitude,
    longitude: loc.longitude,
    water_body: loc.waterBody,
    depth_m: loc.depth,
    captured_at: loc.capturedAt,
    notes: loc.notes,
  };
  Object.entries(fields).forEach(([key, value]) => {
    if (value !== undefined && value !== null && String(value).trim() !== '') body.append(key, String(value).trim());
  });
  body.append('frame_interval', DEFAULT_FRAME_INTERVAL);
  body.append('confidence_threshold', DEFAULT_CONFIDENCE_THRESHOLD);

  const accepted = await requestJson('/api/v1/analyses', { method: 'POST', body, signal });
  onProgress?.(accepted);
  return pollAnalysis(accepted.analysis_id, onProgress, signal);
};

const getFilename = (contentDisposition, fallback) => {
  const match = contentDisposition?.match(/filename\*?=(?:UTF-8'')?['"]?([^;'"\r\n]+)/i);
  return match ? decodeURIComponent(match[1]) : fallback;
};

export const downloadMedia = async (analysis, frame, onStatus, signal) => {
  if (!analysis?.analysis_id || !frame?.id) return false;
  const endpoint = frame.kind === 'video'
    ? `/api/v1/analyses/${analysis.analysis_id}/media/${frame.media_id}/annotated`
    : `/api/v1/analyses/${analysis.analysis_id}/frames/${frame.id}/annotated`;
  const startedAt = Date.now();
  let response;
  while (Date.now() - startedAt < DOWNLOAD_TIMEOUT_MS) {
    response = await fetch(apiUrl(endpoint), { signal });
    if (response.status === 202) {
      const status = await response.json().catch(() => ({}));
      onStatus?.({ loading: true, error: '', progress: status.progress || 0 });
      await new Promise((resolve, reject) => {
        const timer = setTimeout(resolve, 1500);
        signal?.addEventListener('abort', () => { clearTimeout(timer); reject(new DOMException('Download cancelled.', 'AbortError')); }, { once: true });
      });
      continue;
    }
    break;
  }
  if (!response?.ok) throw new Error('Annotated media is not available.');
  const contentType = response.headers.get('content-type') || '';
  const expectedType = frame.kind === 'video' ? 'video/mp4' : 'image/';
  if (!contentType.startsWith(expectedType)) throw new Error(`Unexpected annotated media response (${contentType || 'unknown type'}).`);
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = getFilename(response.headers.get('content-disposition'), `hydrowatch-annotated-${frame.label.replace(/[^a-z0-9]+/gi, '-').toLowerCase()}.${frame.kind === 'video' ? 'mp4' : 'jpg'}`);
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
  onStatus?.({ loading: false, error: '', progress: 0 });
  return true;
};

export const generateReport = async (analysis) => {
  if (!analysis?.analysis_id) return false;
  const report = await requestJson(`/api/v1/analyses/${analysis.analysis_id}/report`);
  const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `hydrowatch-report-${analysis.analysis_id}.json`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
  return true;
};

export const getReport = (analysisId, mediaId, signal) => {
  const suffix = mediaId ? `?media_id=${encodeURIComponent(mediaId)}` : '';
  return requestJson(`/api/v1/analyses/${analysisId}/report${suffix}`, { signal });
};

// With an analysis id, that analysis's location is marked is_current and listed first.
export const getHotspots = async (analysisId, signal) => {
  const suffix = analysisId ? `?analysis_id=${encodeURIComponent(analysisId)}` : '';
  const hotspots = await requestJson(`/api/v1/hotspots${suffix}`, { signal });
  return hotspots.map((hotspot) => ({
    ...hotspot,
    lat: hotspot.latitude,
    lng: hotspot.longitude,
    count: hotspot.debris_count ?? hotspot.count,
    risk: hotspot.risk || 'Moderate',
  }));
};

export const getDashboard = (signal) => requestJson('/api/v1/dashboard', { signal });

export const getHealth = (signal) => requestJson('/api/v1/health', { signal });
