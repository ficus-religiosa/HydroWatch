import { Component, useEffect, useRef, useState } from 'react';
import './App.css';
import Sidebar from './components/Sidebar';
import Navbar from './components/Navbar';
import StatCard from './components/StatCard';
import ChartCard from './components/ChartCard';
import VideoUploader from './components/VideoUploader';
import DetectionTable from './components/DetectionTable';
import SeverityBadge from './components/SeverityBadge';
import HotspotMap from './components/HotspotMap';
import ReportSummary from './components/ReportSummary';
import LoadingState from './components/LoadingState';
import EmptyState from './components/EmptyState';
import {
  pollutionSummary,
  detectionResults,
  reportOverview,
  researchSummary,
  homeCapabilities,
} from './data/mockData';
import { analyzeVideo, downloadMedia, generateReport, getDashboard, getHealth, getHotspots, getReport } from './services/api';
import { emptyLocation } from './services/location';

const getLabel = (item) => typeof item === 'string' ? item : item?.label || '';
const getCount = (item) => typeof item === 'object' ? item?.count : null;

class ErrorBoundary extends Component {
  state = { hasError: false };

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  handleBack = () => {
    this.setState({ hasError: false });
    this.props.onBack();
  };

  render() {
    if (this.state.hasError) {
      return (
        <div className="panel analysis-note">
          <h3>We couldn't display this page</h3>
          <p>Something went wrong while rendering the analysis. Return to the dashboard and try again.</p>
          <button type="button" className="primary-button" onClick={this.handleBack}>Back to dashboard</button>
        </div>
      );
    }
    return this.props.children;
  }
}

const pageTitles = {
  Dashboard: 'Operational overview',
  'Video Analysis': 'Video analysis workspace',
  'Detection Results': 'Detection results',
  'Pollution Analysis': 'Pollution assessment',
  Hotspots: 'Marine hotspot map',
  'Environmental Report': 'Environmental report',
  Research: 'Research & methodology',
};

function App() {
  const [activePage, setActivePage] = useState('Dashboard');
  const [selectedMedia, setSelectedMedia] = useState([]);
  const [location, setLocation] = useState(emptyLocation);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [selectedFrame, setSelectedFrame] = useState(0);
  const [classFilter, setClassFilter] = useState('All classes');
  const [analysisProgress, setAnalysisProgress] = useState({ progress: 0, message: '' });
  const [analysisError, setAnalysisError] = useState('');
  const [analysisHotspots, setAnalysisHotspots] = useState([]);
  const [reportData, setReportData] = useState(null);
  const [pollController, setPollController] = useState(null);
  const [downloadState, setDownloadState] = useState({ loading: false, error: '', progress: 0 });
  const [health, setHealth] = useState(null);
  const [backendError, setBackendError] = useState('');
  const [dashboard, setDashboard] = useState(null);
  const [allHotspots, setAllHotspots] = useState([]);
  const [debrisOnly, setDebrisOnly] = useState(false);
  const downloadController = useRef(null);
  const reportCache = useRef(new Map());
  const hotspotCache = useRef(new Map());

  useEffect(() => {
    const analysisId = analysisResult?.analysis_id;
    const selectedMediaId = analysisResult?.frames?.[selectedFrame]?.media_id;
    if (!analysisId) return undefined;
    const controller = new AbortController();
    const cacheKey = `${analysisId}:${selectedMediaId || ''}`;
    if (hotspotCache.current.has(analysisId)) setAnalysisHotspots(hotspotCache.current.get(analysisId));
    else getHotspots(analysisId, controller.signal)
      .then((items) => { hotspotCache.current.set(analysisId, items); setAnalysisHotspots(items); })
      .catch((error) => { if (error.name !== 'AbortError') setAnalysisHotspots([]); });
    if (reportCache.current.has(cacheKey)) setReportData(reportCache.current.get(cacheKey));
    else getReport(analysisId, selectedMediaId, controller.signal)
      .then((report) => { reportCache.current.set(cacheKey, report); setReportData(report); })
      .catch((error) => { if (error.name !== 'AbortError') setReportData(null); });
    return () => controller.abort();
  }, [analysisResult?.analysis_id, analysisResult?.frames, selectedFrame]);

  const pageSubtitle = backendError
    ? 'Backend offline'
    : health ? `Backend connected - ${health.model.device}${health.model.fp16 ? ', fp16' : ''}` : 'Connecting to backend...';

  useEffect(() => {
    const controller = new AbortController();
    getHealth(controller.signal)
      .then((result) => { setHealth(result); setBackendError(''); })
      .catch((error) => { if (error.name !== 'AbortError') setBackendError(error.message); });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (activePage !== 'Dashboard') return undefined;
    const controller = new AbortController();
    getDashboard(controller.signal)
      .then((result) => { setDashboard(result); setBackendError(''); })
      .catch((error) => { if (error.name !== 'AbortError') setBackendError(error.message); });
    return () => controller.abort();
  }, [activePage, analysisResult?.analysis_id]);

  useEffect(() => {
    if (activePage !== 'Hotspots' || analysisResult) return undefined;
    const controller = new AbortController();
    getHotspots(null, controller.signal)
      .then(setAllHotspots)
      .catch((error) => { if (error.name !== 'AbortError') setAllHotspots([]); });
    return () => controller.abort();
  }, [activePage, analysisResult]);

  const handleFileSelect = (files) => {
    if (!files.length) return;
    setSelectedMedia(files);
    pollController?.abort();
    setPollController(null);
    setAnalysisResult(null);
    setSelectedFrame(0);
  };

  const handleAnalyze = async () => {
    if (!selectedMedia.length) return;
    setIsAnalyzing(true);
    setAnalysisError('');
    setAnalysisProgress({ progress: 0, message: 'Analysis queued' });
    const controller = new AbortController();
    setPollController(controller);
    try {
      const result = await analyzeVideo(selectedMedia, location, (status) => {
        setAnalysisProgress({ progress: status.progress || 0, message: status.message || status.status });
      }, controller.signal);
      setAnalysisResult(result);
    } catch (error) {
      setAnalysisError(error.message);
    } finally {
      setIsAnalyzing(false);
      setPollController(null);
    }
  };

  useEffect(() => () => pollController?.abort(), [pollController]);
  useEffect(() => () => downloadController.current?.abort(), []);

  const handleGenerateReport = () => {
    generateReport(analysisResult).catch((error) => setAnalysisError(error.message));
  };

  const activeAnalysis = analysisResult || {
    ...detectionResults,
    media: [],
    frames: [{ id: 1, label: 'Sample frame', detections: detectionResults.objects }],
    density: pollutionSummary.density,
    location: '',
  };

  const currentFrame = activeAnalysis.frames[selectedFrame] || activeAnalysis.frames[0];
  const isMockInference = activeAnalysis.model_info?.is_mock === true;
  const isDebris = (item) => (item.tier || 'debris') === 'debris';
  const matchesFilter = (object) => (classFilter === 'All classes' || object.label === classFilter) && (!debrisOnly || isDebris(object));
  const visibleObjects = currentFrame.detections.filter(matchesFilter);
  const matchingFrameIndices = activeAnalysis.frames
    .map((frame, index) => ({ frame, index }))
    .filter(({ frame }) => (classFilter === 'All classes' && !debrisOnly) || frame.detections.some(matchesFilter));
  const filterClasses = activeAnalysis.classes.filter((item) => !debrisOnly || typeof item === 'string' || isDebris(item));
  const spots = analysisResult ? analysisHotspots : allHotspots;
  const debrisByClass = (analysisResult?.classes || []).filter(isDebris).map((item) => ({ name: item.label, value: item.count }));
  const noAnalysis = <EmptyState title="No analysis yet" body="Upload photos or video in Video Analysis, then come back here to review the results." />;

  const renderHome = () => (
    <section className="landing-page">
      <div className="hero-panel panel">
        <div className="hero-copy">
          <p className="eyebrow">HydroWatch</p>
          <h2>Turning Underwater Video into Actionable Marine Intelligence</h2>
          <p className="hero-text">
            HydroWatch is a marine debris detection and pollution assessment system designed to
            transform underwater footage into research-grade operational insight for coastal and
            environmental monitoring teams.
          </p>
          <div className="hero-actions">
            <button type="button" className="primary-button" onClick={() => setActivePage('Video Analysis')}>
              Analyze Underwater Video
            </button>
          </div>
          <div className="hero-process">
            <h4>How it works</h4>
            <p>
              Video input is reviewed, debris is classified, hotspot density is mapped, and the system
              summarizes environmental risk for field operations and research reporting.
            </p>
          </div>
        </div>

        <div className="hero-card">
          <div className="mini-stat">
            <span>Live field review</span>
            <strong>84% confidence</strong>
          </div>
          <div className="mini-stat">
            <span>Priority zones</span>
            <strong>14 hotspots</strong>
          </div>
          <div className="mini-stat">
              <span>Review mode</span>
            <strong>Frame review</strong>
          </div>
        </div>
      </div>

      <div className="capability-grid">
        {homeCapabilities.map((item) => (
          <div key={item} className="panel capability-card">
            <div className="capability-dot" />
            <span>{item}</span>
          </div>
        ))}
      </div>
    </section>
  );

  const renderDashboard = () => (
    <div className="page-stack">
      {backendError && <div className="panel analysis-note">{backendError}</div>}
      {dashboard && dashboard.totals.analyses === 0 && (
        <EmptyState title="No analyses yet" body="Upload photos or video in Video Analysis. Totals, recent analyses and the debris mix will appear here." />
      )}
      <div className="stat-grid">
        {(dashboard?.stats || []).map((stat) => (
          <StatCard key={stat.label} {...stat} />
        ))}
      </div>

      <div className="content-grid two-col">
        <ChartCard title="Debris mix across all analyses (%)" data={dashboard?.distribution || []} dataKeys={['value']} color="#5ec3ff" />
        <div className="panel dashboard-note">
          <p className="eyebrow">Review workspace</p>
          <h3>Inspect each capture with model-ready media</h3>
          <p>Upload photos or video clips to review detected debris, frame density, and locations from one workspace.</p>
        </div>
      </div>

      <div className="content-grid two-col">
        <div className="panel">
          <div className="section-heading">
            <h3>Recent analysis</h3>
          </div>
          <div className="timeline-list">
            {(dashboard?.recent || []).map((item) => (
              <div className="timeline-item" key={item.id}>
                <div>
                  <strong>{item.site}</strong>
                  <p>{item.id}</p>
                </div>
                <span>{item.time ? new Date(item.time).toLocaleString() : ''}</span>
                <span className="muted-text">{item.count || 'Media review'}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="panel">
          <div className="section-heading">
            <h3>Recent detection summary</h3>
          </div>
          <div className="summary-list">
            {(dashboard?.distribution || []).map((item) => (
              <div className="summary-row" key={item.name}>
                <span>{item.name}</span>
                <div className="progress-bar">
                  <div className="progress-fill" style={{ width: `${item.value}%` }} />
                </div>
                <strong>{item.value}%</strong>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );

  const renderVideoAnalysis = () => (
    <div className="page-stack">
      <VideoUploader
        mediaFiles={selectedMedia}
        onFileSelect={handleFileSelect}
        onAnalyze={handleAnalyze}
        isAnalyzing={isAnalyzing}
        analysisResult={analysisResult}
        location={location}
        onLocationChange={setLocation}
      />

      {isAnalyzing && <LoadingState text={`${analysisProgress.message} (${analysisProgress.progress}%)`} />}
      {analysisError && <div className="panel analysis-note">{analysisError}</div>}

      {!selectedMedia.length && !isAnalyzing && (
        <EmptyState title="No media uploaded" body="Select one or more photos or video clips to begin detection." />
      )}

      {analysisResult && (
        <div className="content-grid two-col">
          <div className="panel">
            <div className="section-heading">
              <h3>Analysis timeline</h3>
            </div>
            <div className="timeline-tags">
              {matchingFrameIndices.map(({ frame, index }) => (
                <button type="button" key={frame.id} className={`timeline-tag ${selectedFrame === index ? 'selected' : ''}`} onClick={() => setSelectedFrame(index)}>
                  <span>{frame.label}</span>
                  <strong>{frame.label} - {frame.detections.length} objects</strong>
                </button>
              ))}
            </div>
          </div>

          <div className="panel">
            <div className="section-heading">
              <h3>Detected categories</h3>
            </div>
            <ul className="tag-list">
              {analysisResult.classes.map((category) => (
                <li key={getLabel(category)}>{getLabel(category)}{getCount(category) !== null ? ` (${getCount(category)})` : ''}</li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  );

  const renderDetections = () => (!analysisResult ? noAnalysis : (
    <div className="page-stack">
      <div className="stat-grid small-grid">
        <StatCard label="Detected objects" value={activeAnalysis.totalDetections} change="In selected media" tone="blue" />
        <StatCard label="Avg confidence" value={`${activeAnalysis.averageConfidence}%`} change="Model score" tone="teal" />
        <StatCard label="Debris classes" value={activeAnalysis.classes.length} change="Detected classes" tone="amber" />
        <StatCard label="Frames reviewed" value={activeAnalysis.frames.length} change="Photo or video" tone="blue" />
      </div>
      {isMockInference && <p className="eyebrow">Demo / mock inference</p>}

      <div className="content-grid two-col">
        <div className="panel detection-visual-panel">
          <div className="section-heading">
            <h3>Detection visualization</h3>
          </div>
          <div className="detection-frame" style={currentFrame.width && currentFrame.height ? { aspectRatio: `${currentFrame.width} / ${currentFrame.height}`, minHeight: 0 } : undefined}>
            <div className="media-review-frame">
              {currentFrame.mediaUrl ? <img src={currentFrame.mediaUrl} alt={currentFrame.label} /> : <div className="empty-visual">Upload media to preview detections</div>}
            </div>
            {visibleObjects.map((object) => (
              <div
                key={object.id}
                className="detection-box"
                style={{
                  left: `${object.x}%`,
                  top: `${object.y}%`,
                  width: `${object.width}%`,
                  height: `${object.height}%`,
                }}
              >
                <span>{object.label}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="panel">
          <div className="section-heading">
            <h3>Class distribution</h3>
          </div>
          <div className="class-list">
            {activeAnalysis.classes.map((item) => (
              <div key={getLabel(item)} className="class-row">
                <span>{getLabel(item)}</span>
                <strong>{getCount(item) ?? ''}</strong>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="content-grid two-col">
        <div className="panel">
          <div className="section-heading">
            <h3>Size breakdown</h3>
          </div>
          <div className="size-metrics">
            <div><span>Small</span><strong>{activeAnalysis.sizeStats.small}</strong></div>
            <div><span>Medium</span><strong>{activeAnalysis.sizeStats.medium}</strong></div>
            <div><span>Large</span><strong>{activeAnalysis.sizeStats.large}</strong></div>
          </div>
        </div>

        <div className="panel">
          <div className="section-heading">
            <h3>Confidence scores</h3>
          </div>
          <div className="confidence-list">
            {visibleObjects.map((object) => (
              <div key={object.id} className="confidence-row">
                <span>{object.label}</span>
                <strong>{object.confidence}%</strong>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="panel review-controls">
        <label htmlFor="class-filter">Find a debris class</label>
        <select id="class-filter" value={classFilter} onChange={(event) => setClassFilter(event.target.value)}>
          <option>All classes</option>
          {filterClasses.map((item) => <option key={getLabel(item)}>{getLabel(item)}</option>)}
        </select>
        <label className="toggle-label" htmlFor="debris-only">
          <input id="debris-only" type="checkbox" checked={debrisOnly} onChange={(event) => { setDebrisOnly(event.target.checked); setClassFilter('All classes'); }} />
          Debris only (hide fish, invertebrates and plants)
        </label>
        <div className="frame-selector" aria-label="Matching frames">
          {matchingFrameIndices.map(({ frame, index }) => <button type="button" className={`timeline-tag ${selectedFrame === index ? 'selected' : ''}`} key={frame.id} onClick={() => setSelectedFrame(index)}>{index + 1}</button>)}
        </div>
        <button type="button" className="secondary-button" disabled={downloadState.loading} onClick={async () => {
          setDownloadState({ loading: true, error: '' });
          downloadController.current?.abort();
          downloadController.current = new AbortController();
          try { await downloadMedia(activeAnalysis, currentFrame, setDownloadState, downloadController.current.signal); }
          catch (error) { setDownloadState({ loading: false, error: error.message }); }
          finally { downloadController.current = null; }
        }}>{downloadState.loading ? `Preparing annotated ${currentFrame.kind === 'video' ? 'video' : 'photo'}${downloadState.progress ? ` (${downloadState.progress}%)` : ''}...` : `Download annotated ${currentFrame.kind === 'video' ? 'video' : 'photo'}`}</button>
        {downloadState.error && <p className="analysis-note">{downloadState.error}</p>}
      </div>
      <DetectionTable rows={visibleObjects} />
    </div>
  ));

  const renderPollutionAnalysis = () => (!analysisResult ? noAnalysis : (
    <div className="page-stack">
      <div className="panel density-panel">
        <div className="section-heading">
          <p className="eyebrow">Per media measurement</p>
          <h3>Debris density by image or frame</h3>
        </div>
        <div className="summary-grid">
          <div>
            <span>Current frame</span>
            <strong>{currentFrame.label}</strong>
          </div>
          <div>
            <span>Debris in this frame</span>
            <strong>{currentFrame.density || activeAnalysis.density}</strong>
          </div>
          <div>
            <span>Average across frames</span>
            <strong>{activeAnalysis.density}</strong>
          </div>
          <div>
            <span>Busiest frame</span>
            <strong>{activeAnalysis.peakDebris ?? 0} debris</strong>
          </div>
          <div>
            <span>Prototype risk</span>
            <strong><SeverityBadge value={activeAnalysis.risk || 'Low'} /></strong>
          </div>
        </div>
        <p className="analysis-note">Density counts debris detections in each photo or frame (fish, invertebrates and plants are excluded). It is a relative measure: a true items-per-square-metre figure would need the camera's distance to the seabed.</p>
      </div>

      <div className="content-grid two-col">
        <ChartCard title="Debris detected by class" data={debrisByClass} dataKeys={['value']} color="#4ade80" />
        <div className="panel">
          <div className="section-heading"><h3>Density measurements</h3></div>
          <div className="timeline-tags">
            {activeAnalysis.frames.map((frame, index) => <button type="button" className={`timeline-tag ${selectedFrame === index ? 'selected' : ''}`} key={frame.id} onClick={() => setSelectedFrame(index)}>{frame.label}: {frame.density}</button>)}
          </div>
        </div>
      </div>
    </div>
  ));

  const renderHotspots = () => (
    <div className="page-stack">
    {isMockInference && <p className="eyebrow">Demo / mock inference</p>}
      <div className="content-grid two-col map-layout">
        <HotspotMap hotspots={spots} focus={analysisResult?.locationInfo} locationQuery={analysisResult ? analysisResult.location : location.name} />

        <div className="panel hotspot-summary-panel">
          <div className="section-heading">
            <h3>Hotspot summary</h3>
          </div>
          <div className="hotspot-list">
            {!spots.length && <p className="analysis-note">No mapped locations yet. Analyse media with a place name, coordinates or GPS-tagged photos to add one.</p>}
            {spots.map((spot) => (
              <div key={spot.id} className="hotspot-row">
                <div>
                  <strong>{spot.name}</strong>
                  <p>{spot.is_current ? 'This analysis - ' : ''}{spot.analyses_count ?? 1} survey{spot.analyses_count === 1 ? '' : 's'}, {spot.mean_debris_per_frame ?? '-'} debris per frame</p>
                </div>
                <div className="hotspot-meta">
                  <SeverityBadge value={spot.risk} />
                  <span>{spot.count} debris</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );

  const renderReport = () => (!analysisResult ? noAnalysis : (
    <div className="page-stack">
    {reportData?.model_info?.is_mock && <p className="eyebrow">Demo / mock inference</p>}
      <ReportSummary overview={analysisResult ? { ...reportOverview, ...analysisResult, ...(reportData || {}), surveySummary: `Media review for ${analysisResult.media.map((item) => item.name).join(', ')}.` } : reportOverview} />

      <div className="content-grid two-col">
        <div className="panel">
          <div className="section-heading">
            <h3>Environmental impact summary</h3>
          </div>
          <p className="analysis-note">{analysisResult?.location ? `Recorded location: ${analysisResult.location}.` : 'No location was supplied for this media review.'} The report summarizes only the uploaded image or video analysis.</p>
          {analysisResult?.locationInfo && (
            <ul className="bullet-list">
              {analysisResult.locationInfo.latitude != null && <li>Coordinates: {analysisResult.locationInfo.latitude}, {analysisResult.locationInfo.longitude}{analysisResult.locationInfo.source ? ` (${analysisResult.locationInfo.source})` : ''}</li>}
              {analysisResult.locationInfo.water_body && <li>Water body: {analysisResult.locationInfo.water_body}</li>}
              {analysisResult.locationInfo.depth_m != null && <li>Approximate depth: {analysisResult.locationInfo.depth_m} m</li>}
              {analysisResult.locationInfo.captured_at && <li>Date filmed: {analysisResult.locationInfo.captured_at}</li>}
              {analysisResult.locationInfo.notes && <li>Notes: {analysisResult.locationInfo.notes}</li>}
            </ul>
          )}
          {reportData && <p className="analysis-note">Debris detections: {reportData.debris_detections} across {reportData.frames_reviewed} photos/frames ({reportData.density?.value} per frame on average).</p>}
          {reportData?.severity && <div><strong>Prototype severity:</strong> {reportData.severity.category} ({reportData.severity.methodology})</div>}
          {reportData?.impact && <div><strong>Prototype impact:</strong> {reportData.impact.summary} ({reportData.impact.methodology})</div>}
        </div>

        <div className="panel">
          <div className="section-heading">
            <h3>Debris categories</h3>
          </div>
          <ul className="tag-list">
            {(analysisResult?.classes || reportOverview.categories).map((item) => (
              <li key={getLabel(item)}>{getLabel(item)}{getCount(item) !== null ? ` (${getCount(item)})` : ''}</li>
            ))}
          </ul>
        </div>
      </div>

      <div className="report-actions">
        <button type="button" className="primary-button" onClick={() => window.print()}>Print report</button>
        <button type="button" className="secondary-button" onClick={handleGenerateReport}>Download report</button>
      </div>
    </div>
  ));

  const renderResearch = () => (
    <div className="page-stack">
      <div className="panel research-panel">
        <div className="section-heading">
          <h3>{researchSummary.title}</h3>
        </div>
        <p className="analysis-note">{researchSummary.section}</p>

        <div className="content-grid two-col">
          <div>
            <h4>Research motivation</h4>
            <ul className="bullet-list">
              {researchSummary.goals.map((goal) => (
                <li key={goal}>{goal}</li>
              ))}
            </ul>
          </div>

          <div>
            <h4>Dataset sources</h4>
            <ul className="bullet-list">
              {researchSummary.datasetSources.map((source) => (
                <li key={source}>{source}</li>
              ))}
            </ul>
          </div>
        </div>

        <div className="content-grid two-col">
          <div>
            <h4>Model information</h4>
            <p className="analysis-note">{researchSummary.modelInfo}</p>
          </div>

          <div>
            <h4>System workflow</h4>
            <ol className="ordered-list">
              {researchSummary.workflow.map((step) => (
                <li key={step}>{step}</li>
              ))}
            </ol>
          </div>
        </div>
      </div>
    </div>
  );

  const pageMap = {
    Dashboard: renderDashboard,
    'Video Analysis': renderVideoAnalysis,
    'Detection Results': renderDetections,
    'Pollution Analysis': renderPollutionAnalysis,
    Hotspots: renderHotspots,
    'Environmental Report': renderReport,
    Research: renderResearch,
  };

  return (
    <div className="app-shell">
      <Sidebar activePage={activePage} onSelect={setActivePage} />

      <main className="main-panel">
        <Navbar title={activePage === 'Home' ? 'HydroWatch' : pageTitles[activePage]} subtitle={pageSubtitle} />

        <ErrorBoundary key={activePage} onBack={() => setActivePage('Dashboard')}>
          {activePage === 'Home' ? renderHome() : pageMap[activePage]()}
        </ErrorBoundary>
      </main>
    </div>
  );
}

export default App;
