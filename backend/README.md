# HydroWatch backend

A FastAPI server that runs the HydroNet debris detector on uploaded photos and videos, stores results in
MongoDB, and serves them to the website. It implements the API described in `frontend/BACKEND_HANDOFF.md`.

## What it does

1. **Receives uploads** — one or more photos/videos, plus an optional location (place name, coordinates,
   water body, depth, date, notes).
2. **Queues the job** and replies immediately with an `analysis_id`. The website polls for progress.
3. **Runs the detector on the GPU** (fp16), one job at a time so uploads never compete for VRAM:
   - photos: the whole image
   - videos: one frame every `frame_interval` seconds (default 1 s) for the review list
4. **Saves** a preview and an annotated JPEG per photo/frame, and the detections, to disk and MongoDB.
5. **Renders an annotated MP4** afterwards, with boxes on *every* frame (H.264, plays in the browser).
6. **Updates location totals**, which become the hotspot map.

### Locations

The first source available is used:

1. Coordinates the user typed, or shared from the browser
2. The place name, looked up on OpenStreetMap (cached in MongoDB, so each name is looked up once)
3. GPS data inside an uploaded photo (EXIF)

Analyses within ~100 m of each other add up into one hotspot.

### Density and risk — what the numbers mean

- **Density** = debris detections per photo/frame. Fish, invertebrates and plants are excluded.
  It is a *relative* measure. True items per square metre would need the camera's distance to the seabed.
- **Risk** (prototype) comes from the mean debris per frame at a place: Low < 1, Moderate < 3, High < 6,
  Critical ≥ 6. It's useful for comparing places, not a calibrated environmental index.

## Setup (Windows, once)

**1. MongoDB.** Install **MongoDB Community Server** from mongodb.com, keeping "Install as a service"
ticked, so it starts automatically. Nothing else is needed: the `hydrowatch` database and its
collections are created on first start. MongoDB Compass (optional) lets you browse the data.

**2. Model weights.** Copy your trained checkpoint to `backend\weights\best.pt`, for example:
```
copy ..\..\models\hydrowatch_train\runs\oracle_v3_more\weights\best.pt weights\best.pt
```
`hydronet.py` in this folder must stay: the checkpoint needs it to load.

**3. Python packages** — into the same environment you trained in (it already has torch + CUDA):
```
cd backend
pip install -r requirements.txt
```

**4. Settings (optional).** `.env` holds every setting, with comments. The defaults suit a laptop
demo: GPU, fp16, local MongoDB.

## Run

```
cd backend
python run.py
```
Wait for: `HydroWatch ready - model on NVIDIA GeForce RTX 3060 Laptop GPU (fp16=True), database 'hydrowatch'`

Then check:
- `http://127.0.0.1:8000/api/v1/health` — model, device, and queue
- `http://127.0.0.1:8000/docs` — interactive API documentation; you can upload a test file from here

If you've run `npm run build` in `frontend`, the website is also served at `http://127.0.0.1:8000`.

**Don't train and serve at the same time** — both need the GPU's memory.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/health` | Model, device, queue status |
| POST | `/api/v1/analyses` | Upload `files[]`, plus optional `location`, `latitude`, `longitude`, `water_body`, `depth_m`, `captured_at`, `notes`, `frame_interval`, `confidence_threshold`. Returns 202 + `analysis_id` |
| GET | `/api/v1/analyses` | Recent analyses |
| GET | `/api/v1/analyses/{id}` | Status/progress; when completed, includes every frame and its detections |
| GET | `/api/v1/analyses/{id}/frames[/{frame_id}]` | Frames on their own |
| GET | `/api/v1/analyses/{id}/frames/{frame_id}/annotated` | Annotated JPEG download |
| GET | `/api/v1/analyses/{id}/media/{media_id}/annotated` | Annotated MP4 (video) or JPEG (photo). Returns 202 + progress while rendering |
| GET | `/api/v1/analyses/{id}/report[?media_id=]` | Report data: totals, classes, density, severity, impact, location |
| GET | `/api/v1/hotspots[?analysis_id=]` | All mapped locations. With an id, that analysis's location comes first, marked `is_current` |
| GET | `/api/v1/dashboard` | Totals, recent analyses, and the debris mix, for the dashboard |

Boxes are normalised to 0–1 (`bbox.x, y, width, height`), and each frame carries its pixel `width`/`height`.

## Where data lives

| Place | Contents |
|---|---|
| MongoDB `analyses` | One document per upload: status, location, media, summary |
| MongoDB `frames` | One per reviewed photo/frame, with its detections |
| MongoDB `locations` | Running totals per place — the hotspots |
| MongoDB `geocode_cache` | Place-name lookups |
| `storage\analyses\<id>\` | `uploads\` originals · `frames\` previews + annotated JPEGs · `annotated\` MP4s |

**Start fresh:** stop the server, delete the `storage` folder, and drop the `hydrowatch` database
(in Compass, or `mongosh` → `use hydrowatch` → `db.dropDatabase()`).

## Troubleshooting

| Message / symptom | Fix |
|---|---|
| `MongoDB is not reachable` | Start the service: Services app → "MongoDB Server" → Start. Or check `MONGO_URI` |
| `Model weights not found` | Put `best.pt` in `backend\weights\`, or set `MODEL_WEIGHTS` in `.env` |
| `model on cpu` when you expected the GPU | The environment's torch has no CUDA. Use the environment you trained in |
| Out-of-memory on long videos | Set `BATCH=2` in `.env`. Close training runs and other GPU apps |
| Place names never appear on the map | No internet, or OpenStreetMap is rate-limiting. Enter coordinates instead |
| Annotated video won't play in the browser | `imageio-ffmpeg` is missing, so it fell back to a codec browsers can't play. `pip install imageio-ffmpeg`, then re-download |
| Port 8000 already in use | Set `PORT=8001` in `backend\.env` **and** `VITE_API_BASE_URL=http://127.0.0.1:8001` in `frontend\.env` |

## Limits

- Up to 20 files per upload, 500 MB per file, 10-minute videos — all changeable in `.env`.
- There's no login: this is a public, guest-only demo. Don't expose it to the internet as-is.
- Analyses left unfinished by a crash are marked failed on the next start.
