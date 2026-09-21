function SeverityBadge({ value }) {
  const severityMap = {
    Low: 'severity-low',
    Moderate: 'severity-moderate',
    High: 'severity-high',
    Critical: 'severity-critical',
  };

  return <span className={`severity-badge ${severityMap[value] || 'severity-low'}`}>{value}</span>;
}

export default SeverityBadge;
