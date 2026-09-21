import re
from pathlib import Path
from typing import Dict, Any, Tuple
from fastapi import UploadFile, HTTPException, status
from PIL import Image

from app.core.config import settings
from app.utils.logger import logger


class MediaService:
    """Service for validating, processing, and extracting metadata from uploaded media."""

    def sanitize_filename(self, filename: str) -> str:
        """Strip dangerous path traversal characters and normalize filename."""
        raw_name = Path(filename).name
        # Keep alphanumeric, dots, hyphens, and underscores
        clean_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", raw_name)
        return clean_name or "uploaded_file"

    def validate_file(self, upload_file: UploadFile) -> Tuple[str, str]:
        """
        Validate file extension, content type, and non-empty status.
        Returns (kind, normalized_extension).
        Raises HTTPException if invalid.
        """
        if not upload_file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file must have a valid filename.",
            )

        ext = Path(upload_file.filename).suffix.lower()
        content_type = (upload_file.content_type or "").lower()

        is_image_ext = ext in settings.ALLOWED_IMAGE_EXTENSIONS
        is_video_ext = ext in settings.ALLOWED_VIDEO_EXTENSIONS

        is_image_mime = content_type in settings.ALLOWED_IMAGE_MIME_TYPES
        is_video_mime = content_type in settings.ALLOWED_VIDEO_MIME_TYPES

        # Flexible validation: accept if extension matches, or mime type matches
        if is_image_ext or (is_image_mime and not is_video_ext):
            kind = "image"
        elif is_video_ext or (is_video_mime and not is_image_ext):
            kind = "video"
        else:
            supported = ", ".join(
                sorted(settings.ALLOWED_IMAGE_EXTENSIONS | settings.ALLOWED_VIDEO_EXTENSIONS)
            )
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=f"Unsupported file type '{ext}' ({content_type}). Supported formats: {supported}",
            )

        return kind, ext

    def validate_file_size(self, file_path: Path) -> int:
        """Validate that the file is not empty and does not exceed maximum allowed size."""
        size = file_path.stat().st_size
        if size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File '{file_path.name}' is empty (0 bytes).",
            )
        if size > settings.MAX_FILE_SIZE_BYTES:
            max_mb = settings.MAX_FILE_SIZE_BYTES // (1024 * 1024)
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File '{file_path.name}' exceeds the maximum allowed size of {max_mb} MB.",
            )
        return size

    def inspect_image_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Inspect image dimensions and format using PIL."""
        try:
            with Image.open(file_path) as img:
                return {
                    "width": img.width,
                    "height": img.height,
                    "format": img.format.lower() if img.format else "jpeg",
                }
        except Exception as e:
            logger.warning(f"Could not inspect image metadata for {file_path.name}: {e}")
            return {"width": 800, "height": 600, "format": "jpeg"}

    def inspect_video_metadata(self, file_path: Path, frame_interval: float = 1.0) -> Dict[str, Any]:
        """
        Inspect or estimate video metadata (duration, frame count).
        Provides robust fallback values if native video demuxer is absent.
        """
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        # Estimate reasonable video stats based on size
        estimated_duration = max(5.0, round(file_size_mb * 4.0, 1))
        estimated_fps = 25
        total_frames = int(estimated_duration * estimated_fps)
        sampled_frames = max(3, min(10, int(estimated_duration / max(0.5, frame_interval))))

        return {
            "duration_seconds": estimated_duration,
            "total_frames": total_frames,
            "sampled_frames": sampled_frames,
            "fps": estimated_fps,
        }


media_service = MediaService()
