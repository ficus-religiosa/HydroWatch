import { useEffect, useState } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

const riskColors = {
  Low: '#4ade80',
  Moderate: '#fbbf24',
  High: '#f97316',
  Critical: '#f87171',
};

const WORLD_VIEW = [20, 10];

function MapViewport({ lat, lng, zoom }) {
  const map = useMap();

  useEffect(() => {
    map.setView([lat, lng], zoom);
  }, [lat, lng, zoom, map]);

  return null;
}

// focus: the analysed location returned by the backend ({ latitude, longitude, label, source }).
// locationQuery: a place name to look up in the browser only when the backend has no coordinates.
function HotspotMap({ hotspots, focus, locationQuery }) {
  const [lookup, setLookup] = useState(null);
  const [lookupStatus, setLookupStatus] = useState('');
  const hasFocus = focus && focus.latitude != null && focus.longitude != null;
  const query = hasFocus ? '' : (locationQuery || '').trim();

  useEffect(() => {
    if (!query) return undefined;
    const controller = new AbortController();
    fetch(`https://nominatim.openstreetmap.org/search?format=jsonv2&limit=1&q=${encodeURIComponent(query)}`, {
      signal: controller.signal,
      headers: { Accept: 'application/json' },
    })
      .then((response) => response.json())
      .then(([result]) => {
        if (!result) {
          setLookup(null);
          setLookupStatus(`Could not find "${query}". Try a city, region or country.`);
          return;
        }
        setLookup({ lat: Number(result.lat), lng: Number(result.lon), label: result.display_name });
        setLookupStatus(`Showing ${result.display_name}`);
      })
      .catch((error) => {
        if (error.name !== 'AbortError') setLookupStatus('Location lookup is unavailable right now.');
      });
    return () => controller.abort();
  }, [query]);

  const point = hasFocus
    ? { lat: focus.latitude, lng: focus.longitude, label: focus.label }
    : (query ? lookup : null);
  const first = hotspots[0];
  const [lat, lng] = point ? [point.lat, point.lng] : first ? [first.lat, first.lng] : WORLD_VIEW;
  const zoom = point ? 11 : first ? 6 : 2;

  let status;
  if (hasFocus) status = `Showing ${focus.label || 'the analysed location'}${focus.source ? ` (from ${focus.source})` : ''}.`;
  else if (query) status = lookupStatus || `Finding ${query}...`;
  else if (hotspots.length) status = `${hotspots.length} monitored location${hotspots.length === 1 ? '' : 's'}. Circle size grows with debris found.`;
  else status = 'No mapped locations yet. Analyse media with a location to add one.';

  return (
    <div className="panel map-panel">
      <div className="section-heading">
        <h3>Marine pollution hotspots</h3>
        <p className="map-location-status">{status}</p>
      </div>

      <MapContainer center={WORLD_VIEW} zoom={2} scrollWheelZoom className="map-frame">
        <MapViewport lat={lat} lng={lng} zoom={zoom} />
        <TileLayer
          attribution='&copy; OpenStreetMap contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {hotspots.map((hotspot) => (
          <CircleMarker
            key={hotspot.id}
            center={[hotspot.lat, hotspot.lng]}
            radius={8 + Math.min(14, Math.sqrt(hotspot.count || 0))}
            pathOptions={{ color: riskColors[hotspot.risk] || '#5ec3ff', fillColor: riskColors[hotspot.risk] || '#5ec3ff', fillOpacity: 0.7, weight: hotspot.is_current ? 3 : 1 }}
          >
            <Popup>
              <strong>{hotspot.name}</strong><br />
              Risk: {hotspot.risk}<br />
              Debris found: {hotspot.count}
              {hotspot.mean_debris_per_frame != null && <><br />Per frame: {hotspot.mean_debris_per_frame}</>}
              {hotspot.analyses_count != null && <><br />Surveys: {hotspot.analyses_count}</>}
              {hotspot.categories?.length > 0 && <><br />Main debris: {hotspot.categories.join(', ')}</>}
            </Popup>
          </CircleMarker>
        ))}

        {point && (
          <CircleMarker
            center={[point.lat, point.lng]}
            radius={11}
            pathOptions={{ color: '#ffffff', fillColor: '#22d3ee', fillOpacity: 0.95, weight: 3 }}
          >
            <Popup>
              <strong>Uploaded media location</strong><br />
              {point.label}
            </Popup>
          </CircleMarker>
        )}
      </MapContainer>
    </div>
  );
}

export default HotspotMap;
