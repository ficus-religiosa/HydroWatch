"""Settings, read from backend/.env (KEY=VALUE lines) with safe defaults."""
import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent


def _load_env_file(path):
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.split(" #", 1)[0].split("\t#", 1)[0]   # allow comments after a value
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env_file(BACKEND_DIR / ".env")


def _env(key, default):
    return os.environ.get(key, default)


def _path(key, default):
    """Relative paths in .env are relative to the backend folder, wherever the server is started from."""
    p = Path(_env(key, default))
    return p if p.is_absolute() else (BACKEND_DIR / p).resolve()


def _flag(key, default):
    return str(_env(key, default)).lower() in ("1", "true", "yes", "on")


class Settings:
    # model
    MODEL_WEIGHTS = _path("MODEL_WEIGHTS", "weights/best.pt")
    IMGSZ = int(_env("IMGSZ", 1280))
    DEVICE = _env("DEVICE", "auto")                   # auto = first GPU if present, else CPU
    HALF = _flag("HALF", True)                        # fp16 on GPU
    DEFAULT_CONFIDENCE = float(_env("DEFAULT_CONFIDENCE", 0.3))
    BATCH = int(_env("BATCH", 4))                     # frames per GPU call for video
    MODEL_NOTE = _env("MODEL_NOTE", "")               # e.g. the TEST scores of these weights, shown in reports
    # object size bins (sqrt(box area) / image long side), tertiles of the oracle_v3 TEST split
    SIZE_CUTS = (float(_env("SIZE_CUT_SMALL", 0.0410)), float(_env("SIZE_CUT_MEDIUM", 0.0941)))

    # database
    MONGO_URI = _env("MONGO_URI", "mongodb://127.0.0.1:27017")
    MONGO_DB = _env("MONGO_DB", "hydrowatch")

    # files
    STORAGE_DIR = _path("STORAGE_DIR", "storage")
    MAX_FILES = int(_env("MAX_FILES", 20))
    MAX_UPLOAD_MB = int(_env("MAX_UPLOAD_MB", 500))
    MAX_VIDEO_SECONDS = int(_env("MAX_VIDEO_SECONDS", 600))
    PREVIEW_MAX_SIDE = int(_env("PREVIEW_MAX_SIDE", 1600))
    RENDER_VIDEOS = _flag("RENDER_VIDEOS", True)      # pre-render annotated MP4s after each analysis

    # web
    CORS_ORIGINS = [o.strip() for o in _env("CORS_ORIGINS", "*").split(",") if o.strip()]
    FRONTEND_DIST = _path("FRONTEND_DIST", "../frontend/dist")
    GEOCODER = _env("GEOCODER", "nominatim")          # "nominatim" or "off"
    GEOCODER_USER_AGENT = _env("GEOCODER_USER_AGENT", "HydroWatch-student-project/1.0")


settings = Settings()
