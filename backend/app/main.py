"""FastAPI application: loads the model and database on startup, serves the API, media files and
(optionally) the built frontend."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api import router
from .config import settings
from .db import init_db
from .pipeline import start_worker

settings.STORAGE_DIR.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app):
    from .detector import Detector   # imported here so the app module loads fast and errors are clear
    init_db()
    app.state.detector = Detector()
    start_worker(app.state.detector)
    info = app.state.detector.info()
    print(f"HydroWatch ready - model on {info['device']} (fp16={info['fp16']}), database '{settings.MONGO_DB}'")
    yield


app = FastAPI(title="HydroWatch API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"],
                   expose_headers=["Content-Disposition"])
app.include_router(router)
app.mount("/media", StaticFiles(directory=settings.STORAGE_DIR), name="media")
if (settings.FRONTEND_DIST / "index.html").exists():   # after `npm run build`, one server serves everything
    app.mount("/", StaticFiles(directory=settings.FRONTEND_DIST, html=True), name="frontend")
