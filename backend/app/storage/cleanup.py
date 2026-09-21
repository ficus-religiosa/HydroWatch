from datetime import datetime, timedelta, timezone
from pathlib import Path
import shutil

from app.core.config import settings
from app.repository.analysis_repository import AnalysisRepository


def cleanup_expired_storage(repository=None) -> None:
    root = settings.STORAGE_BASE_DIR.resolve()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=settings.RETENTION_HOURS)
    for directory in (settings.UPLOAD_DIR, settings.FRAMES_DIR, settings.ANNOTATED_DIR):
        resolved = directory.resolve()
        if root not in resolved.parents and resolved != root:
            continue
        for child in resolved.iterdir():
            if child.resolve().parent != resolved:
                continue
            modified = datetime.fromtimestamp(child.stat().st_mtime, timezone.utc)
            if repository and child.name in {item.analysis_id for item in repository.list_all() if item.status in {"queued", "processing"}}:
                continue
            if modified < cutoff:
                shutil.rmtree(child, ignore_errors=True) if child.is_dir() else child.unlink(missing_ok=True)


def recover_interrupted_jobs(repository: AnalysisRepository) -> None:
    for analysis in repository.list_all():
        if analysis.status in {"queued", "processing"}:
            analysis.status = "failed"
            analysis.progress = min(analysis.progress, 99)
            analysis.message = "Interrupted by server restart"
            repository.save(analysis)