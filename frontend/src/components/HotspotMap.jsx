import { useEffect, useState } from 'react';
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';

const riskColors = {
  Low: '#4ade80',
  Moderate: '#fbbf24',
  High: '#f97316',
  Critical: '#f87171',
};

const defaultCenter = [36.845, -75.95];

function MapViewport({ center }) {
  const map = useMap();

  useEffect(() => {
    map.setView(center, center === defaultCenter ? 11 : 6);
  }, [center, map]);

  return null;
}

function HotspotMap({ hotspots, location }) {
  const [userLocation, setUserLocation] = useState(null);
  const [locationStatus, setLocationStatus] = useState('');

  useEffect(() => {
    if (!location?.trim()) {
      return undefined;
    }

    const controller = new AbortController();

    // Fallback only: the backend location is primary when it resolves coordinates.
    fetch(`https://nominatim.openstreetmap.org/search?format=jsonv2&limit=1&q=${encodeURIComponent(location)}`, {
      signal: controller.signal,
      headers: { Accept: 'application/json' },
    })
      .then((response) => response.json())
      .then(([result]) => {
        if (!result) {
          setUserLocation(null);
          setLocationStatus(`Could not find "${location}". Try a city, state, or country.`);
          return;
        }

        setUserLocation({ lat: Number(result.lat), lng: Number(result.lon), label: result.display_name });
        setLocationStatus(`Showing user location: ${result.display_name}`);
      })
      .catch((error) => {
        if (error.name !== 'AbortError') setLocationStatus('Location lookup is unavailable. Showing default hotspots.');
      });

    return () => controller.abort();
  }, [location]);

  const resolvedLocation = location?.trim() ? userLocation : null;
  const center = resolvedLocation ? [resolvedLocation.lat, resolvedLocation.lng] : defaultCenter;
  const visibleStatus = location?.trim()
    ? locationStatus || `Finding ${location}...`
    : 'Upload a media file with a location to place it on the map.';

  return (
    <div className="panel map-panel">
      <div className="section-heading">
        <h3>Marine pollution hotspots</h3>
        <p className="map-location-status">{visibleStatus}</p>
      </div>

      <MapContainer center={defaultCenter} zoom={11} scrollWheelZoom className="map-frame">
        <MapViewport center={center} />
        <TileLayer
          attribution='&copy; OpenStreetMap contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {hotspots.map((hotspot) => (
          <CircleMarker
            key={hotspot.id}
            center={[hotspot.lat, hotspot.lng]}
            radius={14}
            pathOptions={{ color: riskColors[hotspot.risk], fillColor: riskColors[hotspot.risk], fillOpacity: 0.7 }}
          >
            <Popup>
              <strong>{hotspot.name}</strong><br />
              Risk: {hotspot.risk}<br />
              Debris: {hotspot.count}
            </Popup>
          </CircleMarker>
        ))}

        {resolvedLocation && (
          <CircleMarker
            center={[resolvedLocation.lat, resolvedLocation.lng]}
            radius={11}
            pathOptions={{ color: '#ffffff', fillColor: '#22d3ee', fillOpacity: 0.95, weight: 3 }}
          >
            <Popup>
              <strong>Uploaded media location</strong><br />
              {resolvedLocation.label}
            </Popup>
          </CircleMarker>
        )}
      </MapContainer>
    </div>
  );
}

export default HotspotMap;
