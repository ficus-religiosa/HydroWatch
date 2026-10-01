import { useState } from 'react';
import { locationProblem } from '../services/location';

const WATER_BODIES = ['Sea / ocean', 'Harbour / port', 'Coastal / beach', 'River / estuary', 'Lake / reservoir', 'Other'];

function VideoUploader({ mediaFiles, onFileSelect, onAnalyze, isAnalyzing, analysisResult, location, onLocationChange }) {
  const [showDetails, setShowDetails] = useState(false);
  const [gpsStatus, setGpsStatus] = useState('');
  const update = (key, value) => onLocationChange({ ...location, [key]: value });
  const problem = locationProblem(location);

  const useMyLocation = () => {
    if (!navigator.geolocation) {
      setGpsStatus('This browser cannot share its location. Enter coordinates instead.');
      return;
    }
    setGpsStatus('Requesting your location...');
    navigator.geolocation.getCurrentPosition(
      (position) => {
        onLocationChange({ ...location, latitude: position.coords.latitude.toFixed(6), longitude: position.coords.longitude.toFixed(6) });
        setGpsStatus(`Coordinates added (accurate to about ${Math.round(position.coords.accuracy)} m).`);
        setShowDetails(true);
      },
      (error) => setGpsStatus(error.code === 1 ? 'Location permission was denied. Enter coordinates instead.' : 'Could not get your location. Enter coordinates instead.'),
      { enableHighAccuracy: true, timeout: 10000 },
    );
  };

  return (
    <div className="panel video-uploader">
      <div className="upload-header">
        <div>
          <p className="eyebrow">Input media</p>
          <h3>Upload images and video clips</h3>
          <p className="analysis-note">Photos are analysed whole; videos are sampled about once per second for review, and the annotated video marks every frame.</p>
        </div>
        <label htmlFor="media-upload" className="upload-label">Choose media</label>
        <input id="media-upload" type="file" accept="image/jpeg,image/png,image/webp,image/bmp,video/mp4,video/webm,video/quicktime,video/x-msvideo,video/x-matroska" onChange={(event) => onFileSelect(Array.from(event.target.files || []))} className="video-input" multiple />
      </div>

      <label className="field-label" htmlFor="media-location">Location name (optional)
        <input id="media-location" type="text" value={location.name} onChange={(event) => update('name', event.target.value)} placeholder="e.g. Lokrum Island, Croatia" />
      </label>
      <p className="field-hint">A place name is looked up on the map. Coordinates and site details make hotspot mapping more precise. Photos that carry GPS data are placed automatically.</p>

      <div className="location-actions">
        <button type="button" className="secondary-button" onClick={useMyLocation}>Use my current location</button>
        <button type="button" className="secondary-button" onClick={() => setShowDetails((open) => !open)} aria-expanded={showDetails}>
          {showDetails ? 'Hide location details' : 'Add location details'}
        </button>
      </div>
      {gpsStatus && <p className="field-hint">{gpsStatus} Only use this if you are at the survey site.</p>}

      {showDetails && (
        <div className="location-details">
          <label className="field-label" htmlFor="loc-lat">Latitude
            <input id="loc-lat" type="text" inputMode="decimal" value={location.latitude} onChange={(event) => update('latitude', event.target.value)} placeholder="e.g. 42.6254" />
          </label>
          <label className="field-label" htmlFor="loc-lng">Longitude
            <input id="loc-lng" type="text" inputMode="decimal" value={location.longitude} onChange={(event) => update('longitude', event.target.value)} placeholder="e.g. 18.1210" />
          </label>
          <label className="field-label" htmlFor="loc-water">Water body
            <select id="loc-water" value={location.waterBody} onChange={(event) => update('waterBody', event.target.value)}>
              <option value="">Not specified</option>
              {WATER_BODIES.map((item) => <option key={item}>{item}</option>)}
            </select>
          </label>
          <label className="field-label" htmlFor="loc-depth">Approximate depth (m)
            <input id="loc-depth" type="text" inputMode="decimal" value={location.depth} onChange={(event) => update('depth', event.target.value)} placeholder="e.g. 8" />
          </label>
          <label className="field-label" htmlFor="loc-date">Date filmed
            <input id="loc-date" type="date" value={location.capturedAt} onChange={(event) => update('capturedAt', event.target.value)} />
          </label>
          <label className="field-label location-notes" htmlFor="loc-notes">Notes
            <textarea id="loc-notes" rows={2} maxLength={1000} value={location.notes} onChange={(event) => update('notes', event.target.value)} placeholder="e.g. near the harbour wall, low visibility" />
          </label>
        </div>
      )}
      {problem && <p className="field-error">{problem}</p>}

      <div className="media-file-list">
        {mediaFiles.length ? mediaFiles.map((file) => (
          <div className="media-file-item" key={`${file.name}-${file.lastModified}`}>
            <span className="media-kind">{file.type.startsWith('video/') ? 'VIDEO' : 'PHOTO'}</span>
            <div><strong>{file.name}</strong><span>{(file.size / 1024 / 1024).toFixed(2)} MB</span></div>
            <span className="normalized-chip">{file.type.startsWith('video/') ? 'Video' : 'Photo'}</span>
          </div>
        )) : <div className="empty-visual">No media selected</div>}
      </div>

      <button type="button" className="primary-button" onClick={onAnalyze} disabled={!mediaFiles.length || isAnalyzing || Boolean(problem)}>{isAnalyzing ? 'Analyzing media...' : 'Analyze selected media'}</button>

      {analysisResult && <div className="analysis-summary"><div className="summary-grid summary-grid-three">
        <div><span>Detected objects</span><strong>{analysisResult.totalDetections}</strong></div>
        <div><span>Frames or photos</span><strong>{analysisResult.frames.length}</strong></div>
        <div><span>Input location</span><strong>{analysisResult.location || 'Not supplied'}</strong></div>
      </div></div>}
    </div>
  );
}

export default VideoUploader;
