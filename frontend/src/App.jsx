import { Component, useEffect, useMemo, useRef, useState } from 'react';
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
  dashboardStats,
  recentAnalysis,
  debrisDistribution,
  pollutionSummary,
  hotspots,
  detectionResults,
  reportOverview,
  researchSummary,
  homeCapabilities,
} from './data/mockData';
import { analyzeVideo, downloadMedia, generateReport, getHotspots, getReport } from './services/api';

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
  const [location, setLocation] = useState('');
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [selectedFrame, setSelectedFrame] = useState(0);
  const [classFilter, setClassFilter] = useState('All classes');
  const [analysisProgress, setAnalysisProgress] = useState({ progress: 0, message: '' });
  const [analysisError, setAnalysisError] = useState('');
  const [analysisHotspots, setAnalysisHotspots] = useState([]);
  const [reportData, setReportData] = useState(null);
  const [pollController, setPollController] = useState(null);
  const [downloadState, setDownloadState] = useState({ loading: false, error: '' });
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

  const pageSubtitle = useMemo(() => {
    return activePage === 'Dashboard'
      ? (analysisResult?.model_info?.is_mock ? 'Backend connected - mock inference' : 'Backend connected')
      : (analysisResult?.model_info?.is_mock ? 'Backend connected - mock inference' : 'Backend connected');
  }, [activePage, analysisResult?.model_info?.is_mock]);

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
  const visibleObjects = currentFrame.detections.filter(
    (object) => classFilter === 'All classes' || object.label === classFilter,
  );
  const matchingFrameIndices = activeAnalysis.frames
    .map((frame, index) => ({ frame, index }))
    .filter(({ frame }) => classFilter === 'All classes' || frame.detections.some((object) => object.label === classFilter));

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
      <div className="stat-grid">
        {dashboardStats.filter((stat) => stat.label !== 'Pollution severity').map((stat) => (
          <StatCard key={stat.label} {...stat} />
        ))}
      </div>

      <div className="content-grid two-col">
        <ChartCard title="Debris category distribution" data={debrisDistribution} dataKeys={['value']} color="#5ec3ff" />
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
            {recentAnalysis.map((item) => (
              <div className="timeline-item" key={item.id}>
                <div>
                  <strong>{item.site}</strong>
                  <p>{item.id}</p>
                </div>
                <span>{item.time}</span>
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
            {debrisDistribution.map((item) => (
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

  const renderDetections = () => (
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
          <div className="detection-frame">
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
          {activeAnalysis.classes.map((item) => <option key={getLabel(item)}>{getLabel(item)}</option>)}
        </select>
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
        }}>{downloadState.loading ? 'Preparing annotated video...' : `Download annotated ${currentFrame.kind === 'video' ? 'video' : 'photo'}`}</button>
        {downloadState.error && <p className="analysis-note">{downloadState.error}</p>}
      </div>
      <DetectionTable rows={visibleObjects} />
    </div>
  );

  const renderPollutionAnalysis = () => (
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
            <span>Debris density</span>
            <strong>{currentFrame.density || activeAnalysis.density}</strong>
          </div>
        </div>
      </div>

      <div className="content-grid two-col">
        <ChartCard title="Debris category distribution" data={debrisDistribution} dataKeys={['value']} color="#4ade80" />
        <div className="panel">
          <div className="section-heading"><h3>Density measurements</h3></div>
          <div className="timeline-tags">
            {activeAnalysis.frames.map((frame, index) => <button type="button" className={`timeline-tag ${selectedFrame === index ? 'selected' : ''}`} key={frame.id} onClick={() => setSelectedFrame(index)}>{frame.label}: {frame.density}</button>)}
          </div>
        </div>
      </div>
    </div>
  );

  const renderHotspots = () => (
    <div className="page-stack">
    {isMockInference && <p className="eyebrow">Demo / mock inference</p>}
      <div className="content-grid two-col map-layout">
        <HotspotMap hotspots={analysisHotspots.length ? analysisHotspots : (analysisResult ? [] : hotspots)} location={analysisResult?.location || location} />

        <div className="panel hotspot-summary-panel">
          <div className="section-heading">
            <h3>Hotspot summary</h3>
          </div>
          <div className="hotspot-list">
            {(analysisHotspots.length ? analysisHotspots : (analysisResult ? [] : hotspots)).map((spot) => (
              <div key={spot.id} className="hotspot-row">
                <div>
                  <strong>{spot.name}</strong>
                  <p>{spot.id}</p>
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

  const renderReport = () => (
    <div className="page-stack">
    {reportData?.model_info?.is_mock && <p className="eyebrow">Demo / mock inference</p>}
      <ReportSummary overview={analysisResult ? { ...reportOverview, ...analysisResult, ...(reportData || {}), surveySummary: `Media review for ${analysisResult.media.map((item) => item.name).join(', ')}.` } : reportOverview} />

      <div className="content-grid two-col">
        <div className="panel">
          <div className="section-heading">
            <h3>Environmental impact summary</h3>
          </div>
          <p className="analysis-note">{analysisResult?.location ? `Recorded location: ${analysisResult.location}.` : 'No location was supplied for this media review.'} The report summarizes only the uploaded image or video analysis.</p>
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
  );

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
