# HydroWatch

Upload underwater photos or video → HydroWatch finds marine debris, shows it frame by frame, measures
how much there is, and maps where it concentrates.

```
HydroWatch/
├── frontend/   React website        (details: frontend/README.md)
└── backend/    FastAPI + detector   (details: backend/README.md)
                  └── MongoDB stores results and locations
```

The detector is **HydroNet** — YOLO11-s with extra small-object detail and water-condition awareness.
It recognises 9 debris types plus fish, invertebrates and plants, and scores mAP50 0.80 on a held-out test set.

## First-time setup

1. **Install MongoDB Community Server** (keep "Install as a service" ticked). The database creates itself.
2. **Copy your trained model** to `backend\weights\best.pt`.
3. **Install packages:**
   ```
   cd backend
   pip install -r requirements.txt
   cd ..\frontend
   npm install
   ```

## Run

**Option A — one window (simplest for a demo):**
```
cd frontend
npm run build
cd ..\backend
python run.py
```
Open **http://127.0.0.1:8000**

**Option B — two windows (while changing the website):**
```
cd backend
python run.py
```
```
cd frontend
npm run dev
```
Open **http://localhost:5173**

The top bar should say **"Backend connected - NVIDIA GeForce RTX 3060 Laptop GPU, fp16"**.
If it says "Backend offline", the backend isn't running, or failed at startup — its window says why.

## Using it

1. **Video Analysis** — choose photos/videos, and optionally add a place name or location details → **Analyze**.
2. **Detection Results** — step through frames, filter by class or debris only, download the annotated photo or video.
3. **Pollution Analysis** — debris per frame and the prototype risk level.
4. **Hotspots** — every analysed location on one map.
5. **Environmental Report** — print, or download as JSON.

A 1-minute video takes roughly 1–2 minutes on the laptop GPU; photos take about a second each.
Don't train a model while the site is running — both need the GPU.
