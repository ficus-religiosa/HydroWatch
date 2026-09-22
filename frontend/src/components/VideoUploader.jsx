function VideoUploader({ mediaFiles, onFileSelect, onAnalyze, isAnalyzing, analysisResult, location, onLocationChange }) {
  return (
    <div className="panel video-uploader">
      <div className="upload-header">
        <div>
          <p className="eyebrow">Input media</p>
          <h3>Upload images and video clips</h3>
          <p className="analysis-note">Photos and videos are normalized to the model input contract before analysis.</p>
        </div>
        <label htmlFor="media-upload" className="upload-label">Choose media</label>
        <input id="media-upload" type="file" accept="image/jpeg,image/png,image/webp,image/bmp,video/mp4,video/webm,video/quicktime,video/x-msvideo" onChange={(event) => onFileSelect(Array.from(event.target.files || []))} className="video-input" multiple />
      </div>

      <label className="field-label" htmlFor="media-location">Location (optional)
        <input id="media-location" type="text" value={location} onChange={(event) => onLocationChange(event.target.value)} placeholder="e.g. North Harbor, pier 4" />
      </label>

      <div className="media-file-list">
        {mediaFiles.length ? mediaFiles.map((file) => (
          <div className="media-file-item" key={`${file.name}-${file.lastModified}`}>
            <span className="media-kind">{file.type.startsWith('video/') ? 'VIDEO' : 'PHOTO'}</span>
            <div><strong>{file.name}</strong><span>{(file.size / 1024 / 1024).toFixed(2)} MB</span></div>
            <span className="normalized-chip">{file.type.startsWith('video/') ? 'MP4 input' : 'JPEG input'}</span>
          </div>
        )) : <div className="empty-visual">No media selected</div>}
      </div>

      <button type="button" className="primary-button" onClick={onAnalyze} disabled={!mediaFiles.length || isAnalyzing}>{isAnalyzing ? 'Analyzing media...' : 'Analyze selected media'}</button>

      {analysisResult && <div className="analysis-summary"><div className="summary-grid summary-grid-three">
        <div><span>Detected objects</span><strong>{analysisResult.totalDetections}</strong></div>
        <div><span>Frames or photos</span><strong>{analysisResult.frames.length}</strong></div>
        <div><span>Input location</span><strong>{analysisResult.location || 'Not supplied'}</strong></div>
      </div></div>}
    </div>
  );
}

export default VideoUploader;
