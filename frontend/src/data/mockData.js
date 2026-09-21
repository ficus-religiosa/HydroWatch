export const dashboardStats = [
  { label: 'Total debris detected', value: '2,438', change: '+12.4%', tone: 'blue' },
  { label: 'Unique debris objects', value: '318', change: '+8.1%', tone: 'teal' },
  { label: 'Pollution severity', value: 'High', change: '+3.2%', tone: 'amber' },
  { label: 'High-risk hotspots', value: '14', change: '+2', tone: 'red' },
];

export const recentAnalysis = [
  { id: 'A-1042', site: 'North Harbor', time: '09:42', severity: 'Critical', score: 86 },
  { id: 'A-1038', site: 'East Breakwater', time: '08:15', severity: 'High', score: 74 },
  { id: 'A-1031', site: 'Pelican Reef', time: '07:58', severity: 'Moderate', score: 59 },
  { id: 'A-1026', site: 'South Channel', time: '06:41', severity: 'Low', score: 31 },
];

export const debrisDistribution = [
  { name: 'Plastic Bottle', value: 34 },
  { name: 'Plastic Bag', value: 22 },
  { name: 'Fishing Net', value: 18 },
  { name: 'Plastic Container', value: 16 },
  { name: 'Other Debris', value: 10 },
];

export const severityTrend = [
  { name: 'Jan', low: 22, moderate: 34, high: 18, critical: 8 },
  { name: 'Feb', low: 25, moderate: 36, high: 24, critical: 10 },
  { name: 'Mar', low: 20, moderate: 41, high: 30, critical: 12 },
  { name: 'Apr', low: 18, moderate: 39, high: 33, critical: 14 },
  { name: 'May', low: 16, moderate: 45, high: 38, critical: 17 },
  { name: 'Jun', low: 14, moderate: 42, high: 41, critical: 19 },
];

export const pollutionSummary = {
  severity: 'High',
  score: 81,
  density: '4.2 items/m²',
  explanation:
    'The observed debris concentration is elevated near the harbor mouth and along the coastal channel, where floating plastics and discarded fishing gear are clustered close to the surface path.',
};

export const hotspots = [
  { id: 'H-01', name: 'North Harbor', lat: 36.849, lng: -75.975, risk: 'High', count: 126 },
  { id: 'H-02', name: 'East Breakwater', lat: 36.821, lng: -75.932, risk: 'Moderate', count: 83 },
  { id: 'H-03', name: 'South Channel', lat: 36.805, lng: -75.907, risk: 'Low', count: 31 },
  { id: 'H-04', name: 'Pelican Reef', lat: 36.886, lng: -75.986, risk: 'Critical', count: 149 },
  { id: 'H-05', name: 'Jetty Mouth', lat: 36.837, lng: -75.949, risk: 'Moderate', count: 74 },
  { id: 'H-06', name: 'Harbor Mouth', lat: 36.853, lng: -75.944, risk: 'High', count: 118 },
];

export const detectionResults = {
  totalDetections: 184,
  averageConfidence: 92.4,
  classes: [
    { label: 'Plastic Bottle', count: 56 },
    { label: 'Plastic Bag', count: 41 },
    { label: 'Fishing Net', count: 33 },
    { label: 'Plastic Container', count: 29 },
    { label: 'Other Debris', count: 25 },
  ],
  sizeStats: { small: 63, medium: 74, large: 47 },
  objects: [
    { id: 'D-1101', label: 'Plastic Bottle', confidence: 96, size: 'Medium', x: 18, y: 32, width: 18, height: 24 },
    { id: 'D-1102', label: 'Fishing Net', confidence: 88, size: 'Large', x: 38, y: 28, width: 24, height: 27 },
    { id: 'D-1103', label: 'Plastic Bag', confidence: 93, size: 'Small', x: 58, y: 46, width: 15, height: 18 },
    { id: 'D-1104', label: 'Plastic Container', confidence: 91, size: 'Medium', x: 71, y: 38, width: 17, height: 21 },
    { id: 'D-1105', label: 'Other Debris', confidence: 86, size: 'Large', x: 49, y: 63, width: 20, height: 23 },
    { id: 'D-1106', label: 'Plastic Bottle', confidence: 94, size: 'Small', x: 12, y: 68, width: 16, height: 17 },
  ],
};

export const reportOverview = {
  surveySummary: 'Survey completed along the inner harbor and reef corridor. Current conditions show moderate to high floating debris density with concentrated hotspots near the breakwater and channel mouth.',
  totalDebris: 2438,
  categories: ['Plastic', 'Fishing gear', 'Packaging waste', 'Other debris'],
  severity: 'High',
  impact: 'Surface debris and entanglement risk remain elevated near fishing access points and the main harbor channel.',
  cleanupPriority: 'Priority 2',
  timestamp: '2026-09-18 09:42 UTC',
};

export const researchSummary = {
  title: 'Project Objective',
  section: 'HydroWatch focuses on marine debris detection and pollution assessment using underwater video analysis to support environmental monitoring and cleanup planning.',
  goals: [
    'Marine debris detection from underwater imagery',
    'Video-based object tracking and temporal review',
    'Pollution severity assessment for field observations',
    'Hotspot localization for operational response planning',
    'Research-ready reporting for environmental monitoring teams',
  ],
  datasetSources: [
    'Public marine monitoring image and video resources',
    'Research institution sample collections',
    'Field survey footage and coastal observation archives',
  ],
  modelInfo: 'Model integration is planned for later stages. The current frontend is designed to support real inference results when the backend and AI pipeline are connected.',
  workflow: [
    'Upload underwater video',
    'Run detection workflow',
    'Review detection summary and confidence scores',
    'Assess pollution severity and risk hotspots',
    'Generate environmental briefing report',
  ],
};

export const homeCapabilities = [
  'Marine Debris Detection',
  'Video Tracking',
  'Pollution Severity Assessment',
  'Hotspot Localization',
  'Environmental Reports',
];
