# HydroWatch frontend

React + Vite dashboard for uploading underwater photos and video, reviewing detected debris frame by
frame, mapping hotspots and printing a report. It talks to the FastAPI backend in `../backend`.

## Run

Development (live reload) — the backend must be running first (`python run.py` in `backend`):
```
cd frontend
npm install        # first time only
npm run dev        # open http://localhost:5173
```

One-server demo — build once, and the backend serves the site itself:
```
npm run build      # then open http://127.0.0.1:8000 while the backend runs
```

Checks: `npm run lint` (project source) and `npm run build`.

## Settings

| File | Used by | Contents |
|---|---|---|
| `.env` | `npm run dev` | `VITE_API_BASE_URL=http://127.0.0.1:8000`, frame interval, confidence threshold |
| `.env.production` | `npm run build` | Empty API URL, so the built site calls the server it was loaded from |

`VITE_DEFAULT_FRAME_INTERVAL` (seconds between reviewed video frames, default 1.0) and
`VITE_DEFAULT_CONFIDENCE_THRESHOLD` (default 0.3) are sent with every analysis.

## Pages and where their data comes from

| Page | Data |
|---|---|
| Dashboard | Live totals, recent analyses and debris mix from `/api/v1/dashboard` |
| Video Analysis | Upload + optional location details; progress while the GPU works |
| Detection Results | Boxes over each photo/frame, class filter, **debris-only toggle**, annotated download |
| Pollution Analysis | Debris per frame, average, busiest frame, prototype risk, debris by class |
| Hotspots | Every mapped location from MongoDB; the current analysis is highlighted |
| Environmental Report | Location details, totals, severity and impact for the selected media; print or download |
| Research | Project, dataset and model description (static text in `src/data/mockData.js`) |

## Changes made when connecting the backend

- **Real data everywhere:** the dashboard, pollution chart and hotspot list no longer show mock numbers.
  Results pages show an empty state until an analysis exists.
- **Location details (optional):** coordinates (typed, or "Use my current location"), water body, depth,
  date filmed, notes — all validated before upload. A place name alone still works.
- **Map:** uses the backend's coordinates first; the browser lookup is only a fallback. It shows all
  monitored places instead of a fixed sample coastline, and circle size grows with debris found.
- **Box alignment fix:** the review panel takes each frame's aspect ratio, so boxes sit exactly on objects.
  Previously they drifted whenever the photo's shape differed from the panel's.
- **Detection table:** the position column showed percentages labelled "px"; it now says %, and there's a
  debris / marine-life column.
- **Honest density:** "debris per frame" instead of "items/m²", which can't be measured without knowing
  the camera's distance to the seabed.
- Download progress for annotated videos; backend status (device, fp16) in the top bar; the unused mock
  API was removed.
