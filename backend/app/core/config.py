from pathlib import Path
from typing import List, Set, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    PROJECT_NAME: str = "HydroWatch Backend"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DEBUG: bool = True
    PUBLIC_BASE_URL: str = "http://127.0.0.1:8000"
    INFERENCE_BACKEND: str = "mock"

    # CORS settings (accepts JSON string or list of origins)
    CORS_ORIGINS: Union[List[str], str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",") if i.strip()]
        elif isinstance(v, str):
            import json
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
        return v if isinstance(v, list) else []

    # Upload validation settings
    MAX_FILE_SIZE_BYTES: int = 100 * 1024 * 1024  # 100 MB per file
    MAX_TOTAL_FILES: int = 20
    MAX_VIDEO_DURATION_SECONDS: float = 600.0
    MAX_FRAMES_PER_VIDEO: int = 1000
    FRAME_INTERVAL_DEFAULT_SECONDS: float = 1.0
    FRAME_INTERVAL_MIN_SECONDS: float = 0.1
    FRAME_INTERVAL_MAX_SECONDS: float = 60.0
    CONFIDENCE_THRESHOLD_DEFAULT: float = 0.5
    SMALL_BBOX_AREA_THRESHOLD: float = 0.03
    LARGE_BBOX_AREA_THRESHOLD: float = 0.15
    RETENTION_HOURS: float = 168.0
    CLEANUP_INTERVAL_SECONDS: float = 3600.0
    PROTOTYPE_SEVERITY_MODERATE_COUNT: int = 5
    PROTOTYPE_SEVERITY_HIGH_COUNT: int = 20
    PROTOTYPE_SEVERITY_CRITICAL_COUNT: int = 40
    PROTOTYPE_SEVERITY_SCORE_WEIGHT_COUNT: int = 2
    PROTOTYPE_IMPACT_SUMMARY: str = "Potential environmental concern indicated by detected debris; field validation is required."

    ALLOWED_IMAGE_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    ALLOWED_VIDEO_EXTENSIONS: Set[str] = {".mp4", ".webm", ".mov", ".avi"}

    ALLOWED_IMAGE_MIME_TYPES: Set[str] = {
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/bmp",
        "image/x-ms-bmp",
    }
    ALLOWED_VIDEO_MIME_TYPES: Set[str] = {
        "video/mp4",
        "video/webm",
        "video/quicktime",
        "video/x-msvideo",
        "video/avi",
        "video/x-matroska",
    }

    # Storage paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    STORAGE_BASE_DIR: Path = BASE_DIR / "data"

    @property
    def UPLOAD_DIR(self) -> Path:
        return self.STORAGE_BASE_DIR / "uploads"

    @property
    def FRAMES_DIR(self) -> Path:
        return self.STORAGE_BASE_DIR / "frames"

    @property
    def ANNOTATED_DIR(self) -> Path:
        return self.STORAGE_BASE_DIR / "annotated"

    @property
    def REPORTS_DIR(self) -> Path:
        return self.STORAGE_BASE_DIR / "reports"

    def ensure_storage_directories(self) -> None:
        """Create storage directories if they do not exist."""
        for path in [
            self.STORAGE_BASE_DIR,
            self.UPLOAD_DIR,
            self.FRAMES_DIR,
            self.ANNOTATED_DIR,
            self.REPORTS_DIR,
        ]:
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
