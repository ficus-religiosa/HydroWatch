function ReportSummary({ overview }) {
  return (
    <div className="panel report-summary printable-report">
      <div className="section-heading">
        <p className="eyebrow">HydroWatch media report</p>
        <h3>Analysis of selected media</h3>
      </div>
      <p>{overview.surveySummary}</p>
      <div className="summary-grid report-grid summary-grid-three">
        <div><span>Detected objects</span><strong>{overview.totalDetections || overview.totalDebris || 0}</strong></div>
        <div><span>Frames or photos</span><strong>{overview.frames?.length || 0}</strong></div>
        <div><span>Analysis time</span><strong>{overview.timestamp || 'Current session'}</strong></div>
      </div>
    </div>
  );
}

export default ReportSummary;
