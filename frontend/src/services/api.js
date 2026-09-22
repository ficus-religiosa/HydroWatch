const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '');
const POLL_INTERVAL_MS = 500;
const POLL_TIMEOUT_MS = 10 * 60 * 1000;
const DEFAULT_FRAME_INTERVAL = import.meta.env.VITE_DEFAULT_FRAME_INTERVAL || '1.0';
const DEFAULT_CONFIDENCE_THRESHOLD = import.meta.env.VITE_DEFAULT_CONFIDENCE_THRESHOLD || '0.5';

const apiUrl = (path) => `${API_BASE_URL}${path}`;

const requestJson = async (path, options = {}) => {
  const response = await fetch(apiUrl(path), options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload?.error?.message || payload?.detail || 'HydroWatch request failed.');
  }
  return payload;
};

const densityLabel = (density) => {
  if (!density) return '0.0 items/frame';
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
  averageConfidence: analysis.average_confidence,
  sizeStats: analysis.size_stats,
  density: densityLabel(analysis.density),
  frames: (analysis.frames || []).map(adaptFrame),
  location: analysis.location?.label || analysis.location?.query || '',
});

const pollAnalysis = async (analysisId, onProgress, signal) => {
  const startedAt = Date.now();
  while (Date.now() - startedAt < POLL_TIMEOUT_MS) {
    if (signal?.aborted) throw new DOMException('Analysis polling cancelled.', 'AbortError');
    const analysis = await requestJson(`/api/v1/analyses/${analysisId}`);
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

export const analyzeVideo = async (files, location, onProgress, signal) => {
  const body = new FormData();
  files.forEach((file) => body.append('files', file));
  if (location) body.append('location', location);
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
  while (Date.now() - startedAt < 2 * 60 * 1000) {
    response = await fetch(apiUrl(endpoint), { signal });
    if (response.status === 202) {
      onStatus?.({ loading: true, error: '' });
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
  onStatus?.({ loading: false, error: '' });
  return true;
};

export const generateReport = async (analysis) => {
  if (!analysis?.analysis_id) return false;
  const report = await requestJson(`/api/v1/analyses/${analysis.analysis_id}/report`);
  const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'hydrowatch-report.json';
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

export const getHotspots = async (analysisId, signal) => {
  const hotspots = await requestJson(`/api/v1/hotspots?analysis_id=${encodeURIComponent(analysisId)}`, { signal });
  return hotspots.map((hotspot) => ({
    ...hotspot,
    lat: hotspot.latitude,
    lng: hotspot.longitude,
    count: hotspot.debris_count ?? hotspot.count,
    risk: hotspot.risk || 'Moderate',
  }));
};
