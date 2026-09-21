from contextlib import asynccontextmanager
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path

from app.core.config import settings
from app.api.v1.router import api_router
from app.utils.logger import logger
from app.services.analysis_service import analysis_service
from app.storage.cleanup import cleanup_expired_storage, recover_interrupted_jobs
from app.core.auth import get_current_client
import threading
import time


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Ensure storage folders exist
    logger.info("Initializing HydroWatch Backend...")
    settings.ensure_storage_directories()
    cleanup_expired_storage(analysis_service.repository)
    recover_interrupted_jobs(analysis_service.repository)
    stop_cleanup = threading.Event()
    def cleanup_loop():
        while not stop_cleanup.wait(settings.CLEANUP_INTERVAL_SECONDS):
            cleanup_expired_storage(analysis_service.repository)
    cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
    cleanup_thread.start()
    logger.info(f"Storage directories initialized under: {settings.STORAGE_BASE_DIR}")
    yield
    stop_cleanup.set()
    # Shutdown: Clean up any resources
    logger.info("Shutting down HydroWatch Backend...")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="FastAPI Backend for HydroWatch Marine Debris Detection and Pollution Assessment.",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
    lifespan=lifespan,
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if isinstance(exc.detail, dict) and "code" in exc.detail and "message" in exc.detail:
        error = exc.detail
    else:
        detail = exc.detail if isinstance(exc.detail, str) else "Request failed."
        error = {"code": f"HTTP_{exc.status_code}", "message": detail}
    return JSONResponse(status_code=exc.status_code, content={"error": error})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"error": {"code": "VALIDATION_ERROR", "message": "Request validation failed."}})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled request error")
    return JSONResponse(status_code=500, content={"error": {"code": "INTERNAL_ERROR", "message": "The server could not complete the request."}})

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS if isinstance(settings.CORS_ORIGINS, list) else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(api_router, prefix=settings.API_V1_STR)

# Serve previews only from the configured storage root.
settings.ensure_storage_directories()


@app.get("/static/media/{media_path:path}", dependencies=[Depends(get_current_client)])
async def get_media_file(media_path: str):
    if ".." in Path(media_path).parts or Path(media_path).is_absolute():
        raise HTTPException(status_code=404, detail="Media file not found.")
    candidate = (settings.STORAGE_BASE_DIR / media_path).resolve()
    if settings.STORAGE_BASE_DIR.resolve() not in candidate.parents or not candidate.is_file():
        raise HTTPException(status_code=404, detail="Media file not found.")
    return FileResponse(candidate)


@app.get("/", tags=["Root"])
async def root():
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "online",
        "docs_url": f"{settings.API_V1_STR}/docs",
        "health_check": f"{settings.API_V1_STR}/health",
    }
