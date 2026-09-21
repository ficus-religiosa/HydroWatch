import { detectionResults } from '../data/mockData';

const makeFrameDetections = (seed) => detectionResults.objects
  .filter((_, index) => (index + seed) % 3 !== 0)
  .map((object, index) => ({ ...object, id: `${object.id}-${seed}-${index}` }));

export const analyzeVideoMock = (files, location) => new Promise((resolve) => {
  setTimeout(() => {
    const frames = files.flatMap((file, fileIndex) => {
      const kind = file.type.startsWith('video/') ? 'video' : 'image';
      const frameCount = kind === 'video' ? 5 : 1;
      const mediaUrl = URL.createObjectURL(file);
      return Array.from({ length: frameCount }, (_, frameIndex) => ({
        id: `${file.name}-${frameIndex}`,
        label: kind === 'video' ? `${file.name} / frame ${frameIndex + 1}` : file.name,
        kind,
        mediaUrl,
        density: `${(1.2 + ((fileIndex + frameIndex) % 5) * 0.7).toFixed(1)} items/m²`,
        detections: makeFrameDetections(fileIndex + frameIndex),
      }));
    });

    const allObjects = frames.flatMap((frame) => frame.detections);
    const classMap = allObjects.reduce((map, object) => {
      map[object.label] = (map[object.label] || 0) + 1;
      return map;
    }, {});

    resolve({
      media: files.map((file) => ({ name: file.name, kind: file.type.startsWith('video/') ? 'video' : 'image' })),
      location,
      totalDetections: allObjects.length,
      averageConfidence: allObjects.length ? Number((allObjects.reduce((sum, object) => sum + object.confidence, 0) / allObjects.length).toFixed(1)) : 0,
      classes: Object.entries(classMap).map(([label, count]) => ({ label, count })),
      sizeStats: allObjects.reduce((sizes, object) => ({ ...sizes, [object.size.toLowerCase()]: sizes[object.size.toLowerCase()] + 1 }), { small: 0, medium: 0, large: 0 }),
      density: frames[0]?.density || '0.0 items/m²',
      frames,
    });
  }, 700);
});

export const downloadMediaMock = (analysis, frame) => {
  if (!frame?.mediaUrl) return false;
  const link = document.createElement('a');
  link.href = frame.mediaUrl;
  link.download = `hydrowatch-annotated-${frame.label.replace(/[^a-z0-9]+/gi, '-').toLowerCase()}`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  return true;
};

export const generateReportMock = (analysis, location) => {
  const content = `HydroWatch Media Analysis Report\n\nMedia: ${(analysis?.media || []).map((item) => item.name).join(', ') || 'No media selected'}\nLocation: ${location || 'Not supplied'}\nDetected objects: ${analysis?.totalDetections || 0}\nFrames or photos reviewed: ${analysis?.frames?.length || 0}\nDebris classes: ${(analysis?.classes || []).map((item) => `${item.label} (${item.count})`).join(', ')}\n\nThis report describes the selected media only.`;
  const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = 'hydrowatch-media-report.txt';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
  return true;
};
