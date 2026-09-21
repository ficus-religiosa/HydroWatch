# HydroWatch Frontend to Backend Handoff

## Purpose

The frontend is a React/Vite dashboard for marine debris analysis. It currently uses mock data so the complete user workflow can be demonstrated before backend inference is connected.

The backend should preserve the response shapes described below, or provide an adapter in `frontend/src/services/mockApi.js` / a replacement API service.

## Current Frontend Status

Implemented in the frontend:

- Dashboard with debris category distribution and recent analysis summary.
- Video Analysis workspace.
- Multiple image and video file selection.
- Supported input formats:
  - Images: JPEG, PNG, WebP, BMP.
  - Video: MP4, WebM, MOV/QuickTime, AVI.
- Optional user-provided location, for example `India, Gujarat`.
- Detection Results page with:
  - Detected-object count.
  - Average confidence.
  - Detected debris classes.
  - Frame/photo navigation.
  - Class filtering.
  - Bounding-box overlay area.
  - Size and confidence summaries.
  - Detection table.
- Pollution Analysis page with debris density for the selected image/frame.
- Hotspots page with existing hotspot data and user-location map marker.
- Environmental Report page focused on the selected media, not global pollution severity.
- Browser print action for the environmental report.
- Research page structure preserved for later text updates.

## Important Current Limitations

These are frontend placeholders and should be replaced by backend behavior:

- `frontend/src/services/mockApi.js` fabricates detections, frame counts, confidence, and density.
- Video analysis currently creates five simulated frames per video. The backend must extract real frames or provide real frame metadata.
- Current media download downloads the original browser file URL. It does not yet generate a truly annotated video/photo. The backend should return an annotated-media URL or file.
- Location lookup currently calls the public OpenStreetMap Nominatim service directly from the browser. Prefer backend geocoding or return coordinates from the analysis API.
- Existing dashboard, hotspot, and research sample values are still mock/static data.

## Expected Analysis Request

Recommended endpoint:

```http
POST /api/v1/analyses
Content-Type: multipart/form-data
```

Form fields:

| Field | Type | Required | Description |
|---|---|---:|---|
| `files` | file[] | yes | One or more image/video files. |
| `location` | string | no | Free-text location entered by the user, e.g. `India, Gujarat`. |
| `frame_interval` | number | no | Video sampling interval in seconds or frames. |
| `confidence_threshold` | number | no | Optional model confidence threshold. |

The backend should validate file type and size, normalize media for model input, and return an analysis ID immediately if inference is asynchronous.

Example response for synchronous analysis:

```json
{
  "analysis_id": "analysis_01J...",
  "status": "completed",
  "location": {
    "query": "India, Gujarat",
    "label": "Gujarat, India",
    "latitude": 22.2587,
    "longitude": 71.1924
  },
  "media": [
    {
      "id": "media_001",
      "name": "underwater-survey.mp4",
      "kind": "video",
      "original_format": "video/quicktime",
      "model_format": "mp4",
      "duration_seconds": 42.4,
      "frame_count": 212
    }
  ],
  "total_detections": 18,
  "average_confidence": 91.6,
  "classes": [
    { "label": "Plastic Bottle", "count": 8 },
    { "label": "Fishing Net", "count": 6 }
  ],
  "size_stats": { "small": 4, "medium": 9, "large": 5 },
  "frames": []
}
```

## Frame and Detection Contract

The frontend expects a `frames` array. A photo is represented as one frame. A video contains one entry per reviewable frame or sampled timestamp.

```json
{
  "id": "media_001-frame-0042",
  "media_id": "media_001",
  "frame_number": 42,
  "timestamp_seconds": 8.4,
  "label": "underwater-survey.mp4 / frame 42",
  "kind": "video",
  "density": {
    "value": 3.4,
    "unit": "items/m2"
  },
  "preview_url": "https://api.example.com/media/.../frames/42.jpg",
  "annotated_preview_url": "https://api.example.com/media/.../frames/42-annotated.jpg",
  "detections": [
    {
      "id": "det_0042_01",
      "label": "Plastic Bottle",
      "confidence": 0.96,
      "size": "medium",
      "bbox": {
        "x": 0.18,
        "y": 0.32,
        "width": 0.18,
        "height": 0.24
      }
    }
  ]
}
```

Bounding-box coordinates should preferably be normalized from `0` to `1`. If pixel coordinates are returned, include the source `width` and `height` so the frontend can scale them reliably.

The frontend needs these behaviors from the response:

- Only detections present in the selected input are included.
- A class filter can identify every frame containing that class.
- Each selected frame can render its own detections.
- Density is available per photo/frame, not only as one global value.

## Annotated Media Downloads

Recommended endpoints:

```http
GET /api/v1/analyses/{analysis_id}/media/{media_id}/annotated
GET /api/v1/analyses/{analysis_id}/frames/{frame_id}/annotated
```

Expected behavior:

- For an image, return an annotated JPEG/PNG.
- For a video, return an annotated MP4 with bounding boxes drawn on every requested frame.
- Set the correct `Content-Type` and `Content-Disposition` filename.
- The frontend can use a returned URL or a Blob response for download.

## Location and Hotspots

The user enters location during media upload. The frontend passes the same value into the analysis request and displays the returned coordinates on the Hotspots map.

Preferred backend behavior:

1. Geocode the free-text location server-side.
2. Return canonical label, latitude, and longitude in the analysis response.
3. Associate detections/density with the analysis location.
4. Expose hotspot data through an endpoint such as:

```http
GET /api/v1/hotspots?analysis_id=analysis_01J...
```

Example hotspot object:

```json
{
  "id": "hotspot_001",
  "name": "Gujarat survey point",
  "latitude": 22.2587,
  "longitude": 71.1924,
  "debris_count": 18,
  "categories": ["Plastic Bottle", "Fishing Net"]
}
```

The current frontend uses the public Nominatim service as a temporary fallback. A production backend should avoid depending on browser-side public geocoding and should handle rate limits, caching, and invalid locations.

## Suggested API Surface

```http
POST /api/v1/analyses
GET  /api/v1/analyses/{analysis_id}
GET  /api/v1/analyses/{analysis_id}/frames
GET  /api/v1/analyses/{analysis_id}/frames/{frame_id}
GET  /api/v1/analyses/{analysis_id}/media/{media_id}/annotated
GET  /api/v1/hotspots?analysis_id={analysis_id}
GET  /api/v1/analyses/{analysis_id}/report
```

For long-running videos, use:

```json
{
  "analysis_id": "analysis_01J...",
  "status": "queued",
  "progress": 35,
  "message": "Processing frame 74 of 212"
}
```

The frontend can poll the analysis endpoint or consume a WebSocket/SSE progress stream.

## Frontend Integration Points

Replace these mock functions with real API calls:

- `analyzeVideoMock(files, location)` in `frontend/src/services/mockApi.js`
  - Replace with multipart upload and analysis polling.
- `downloadMediaMock(analysis, frame)`
  - Replace with download of the backend annotated-media endpoint.
- `generateReportMock(analysis, location)`
  - Replace with backend report generation/download if a server-generated PDF is required.

The main state and page routing are in `frontend/src/App.jsx`. The primary visual components are:

- `frontend/src/components/VideoUploader.jsx`
- `frontend/src/components/HotspotMap.jsx`
- `frontend/src/components/DetectionTable.jsx`
- `frontend/src/components/ReportSummary.jsx`

## Backend Acceptance Checklist

- [ ] Accept multiple image/video files in one request.
- [ ] Validate and normalize supported file formats.
- [ ] Return an analysis ID and processing status.
- [ ] Return only detections found in submitted media.
- [ ] Return per-frame/per-image detections and density.
- [ ] Return normalized bounding boxes and confidence values.
- [ ] Return frame preview URLs.
- [ ] Return annotated image/video download URLs.
- [ ] Geocode and return the user-provided location.
- [ ] Return hotspot coordinates and debris counts.
- [ ] Support report generation or provide data for frontend printing.
- [ ] Configure CORS for the Vite development origin, normally `http://localhost:5173`.
- [ ] Protect upload and analysis endpoints with authentication before production use.
- [ ] Enforce upload size, duration, and processing limits.

## Running the Current Frontend

```powershell
cd frontend
npm install
npm run dev
```

Production verification:

```powershell
npm run lint
npm run build
```
