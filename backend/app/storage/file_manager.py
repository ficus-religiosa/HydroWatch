from pathlib import Path
import shutil
from contextvars import ContextVar
from fastapi import UploadFile
from app.core.config import settings
from app.utils.logger import logger

request_base_url: ContextVar[str | None] = ContextVar("request_base_url", default=None)


class FileManager:
    def __init__(self):
        settings.ensure_storage_directories()

    def get_analysis_upload_dir(self, analysis_id: str) -> Path:
        path = settings.UPLOAD_DIR / analysis_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_analysis_frames_dir(self, analysis_id: str) -> Path:
        path = settings.FRAMES_DIR / analysis_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_analysis_annotated_dir(self, analysis_id: str) -> Path:
        path = settings.ANNOTATED_DIR / analysis_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def get_analysis_reports_dir(self, analysis_id: str) -> Path:
        path = settings.REPORTS_DIR / analysis_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_upload_file(self, upload_file: UploadFile, destination_path: Path) -> Path:
        """Stream uploaded file to destination path without third-party dependencies."""
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        with open(destination_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
        upload_file.file.seek(0)
        logger.info(f"Saved uploaded file to {destination_path}")
        return destination_path

    def get_relative_url(self, file_path: Path) -> str:
        """Compute the accessible static URL for a path stored under STORAGE_BASE_DIR."""
        try:
            rel = file_path.relative_to(settings.STORAGE_BASE_DIR)
            base_url = settings.PUBLIC_BASE_URL or request_base_url.get() or ""
            return f"{base_url.rstrip('/')}/static/media/{rel.as_posix()}"
        except ValueError:
            base_url = settings.PUBLIC_BASE_URL or request_base_url.get() or ""
            return f"{base_url.rstrip('/')}/static/media/{file_path.name}"


file_manager = FileManager()
