export const emptyLocation = { name: '', latitude: '', longitude: '', waterBody: '', depth: '', capturedAt: '', notes: '' };

export const locationProblem = (location) => {
  const lat = String(location.latitude ?? '').trim();
  const lng = String(location.longitude ?? '').trim();
  if (!lat && !lng) return '';
  if (!lat || !lng) return 'Enter both latitude and longitude, or leave both empty.';
  if (Number.isNaN(Number(lat)) || Math.abs(Number(lat)) > 90) return 'Latitude must be a number between -90 and 90.';
  if (Number.isNaN(Number(lng)) || Math.abs(Number(lng)) > 180) return 'Longitude must be a number between -180 and 180.';
  const depth = String(location.depth ?? '').trim();
  if (depth && (Number.isNaN(Number(depth)) || Number(depth) < 0)) return 'Depth must be a positive number of metres.';
  return '';
};
