from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import re
import shutil
import subprocess
from fastapi import HTTPException, UploadFile, status
from PIL import Image, ImageOps, UnidentifiedImageError

from app.core.config import settings
from app.storage.file_manager import file_manager
from app.utils.logger import logger


@dataclass
class NormalizedImageResult:
    width: int
    height: int
    format: str
    normalized_path: Path
    relative_url: str


@dataclass
class ExtractedFrame:
    frame_number: int
    timestamp_seconds: float
    path: Path
    relative_url: str


@dataclass
class NormalizedVideoResult:
    width: int
    height: int
    duration: float
    frame_count: int
    fps: float
    normalized_format: str
    rotation: int
    extracted_frames: List[ExtractedFrame]


class MediaValidationNormalizationService:
    """
    Dedicated service for validating media format and integrity,
    extracting technical metadata, and normalizing media for AI processing.
    Completely decoupled from AI inference.
    """

    def validate_file_metadata(self, upload_file: UploadFile) -> Tuple[str, str]:
        """
        Validate file extension and MIME type against supported image/video catalogs.
        Returns (kind, normalized_extension).
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

        if is_image_ext or (is_image_mime and not is_video_ext):
            return "image", ext
        elif is_video_ext or (is_video_mime and not is_image_ext):
            return "video", ext
        else:
            supported_images = ", ".join(sorted(settings.ALLOWED_IMAGE_EXTENSIONS))
            supported_videos = ", ".join(sorted(settings.ALLOWED_VIDEO_EXTENSIONS))
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail=(
                    f"Unsupported file format '{ext}' (MIME: '{content_type}'). "
                    f"Supported image formats: {supported_images}. "
                    f"Supported video formats: {supported_videos}."
                ),
            )

    def validate_file_size(self, file_path: Path) -> int:
        """Validate that file exists, is non-empty, and does not exceed file size cap."""
        if not file_path.exists():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Uploaded file '{file_path.name}' could not be located on disk.",
            )

        size = file_path.stat().st_size
        if size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File '{file_path.name}' is empty (0 bytes) and cannot be processed.",
            )

        if size > settings.MAX_FILE_SIZE_BYTES:
            max_mb = settings.MAX_FILE_SIZE_BYTES // (1024 * 1024)
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File '{file_path.name}' exceeds the maximum allowed upload limit of {max_mb} MB.",
            )

        return size

    def validate_and_normalize_image(
        self, file_path: Path, output_dir: Path, media_id: str
    ) -> NormalizedImageResult:
        """
        Validate image integrity (detect corrupt/unreadable files),
        extract metadata (width, height, format),
        and normalize into a clean RGB JPEG format for the AI pipeline.
        """
        # Step 1: Verify header integrity
        try:
            with Image.open(file_path) as test_img:
                raw_format = (test_img.format or "jpeg").lower()
                test_img.verify()
        except HTTPException:
            raise
        except (UnidentifiedImageError, OSError, ValueError, SyntaxError, Exception) as e:
            logger.error(f"Image integrity verification failed for '{file_path.name}': {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image file '{file_path.name}' is corrupted or unreadable: {str(e)}",
            )

        # Step 2: Re-open and fully decode pixel buffer to detect truncated streams
        try:
            with Image.open(file_path) as img:
                # Force full pixel load to catch truncated image data
                img.load()

                # Correct EXIF orientation
                normalized_img = ImageOps.exif_transpose(img)
                if normalized_img is None:
                    normalized_img = img

                # Normalize color space to RGB (handles RGBA, Palette, CMYK, Grayscale)
                if normalized_img.mode != "RGB":
                    normalized_img = normalized_img.convert("RGB")

                width, height = normalized_img.size

                # Step 3: Save normalized image for AI model consumption
                output_dir.mkdir(parents=True, exist_ok=True)
                normalized_filename = f"{media_id}_normalized.jpg"
                normalized_path = output_dir / normalized_filename
                normalized_img.save(normalized_path, "JPEG", quality=95, optimize=True)

                relative_url = file_manager.get_relative_url(normalized_path)

                return NormalizedImageResult(
                    width=width,
                    height=height,
                    format=raw_format,
                    normalized_path=normalized_path,
                    relative_url=relative_url,
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Image decompression/normalization failed for '{file_path.name}': {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Image file '{file_path.name}' has unreadable or corrupt pixel data: {str(e)}",
            )

    def validate_and_normalize_video(
        self, file_path: Path, output_dir: Path, media_id: str, frame_interval: float = 1.0
    ) -> NormalizedVideoResult:
        """
        Validate video integrity using OpenCV (detect corrupt/unreadable files),
        extract metadata (duration, frame count, width, height, FPS, normalized format),
        and extract normalized review frames for the AI pipeline.
        """
        cap = cv2.VideoCapture(str(file_path))

        try:
            if not cap.isOpened():
                logger.error(f"OpenCV could not open video container for '{file_path.name}'.")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Video file '{file_path.name}' is corrupted or container is unreadable.",
                )

            # Test first frame to verify decodable video stream
            ret, first_frame = cap.read()
            if not ret or first_frame is None or first_frame.size == 0:
                logger.error(f"No decodable frames found in '{file_path.name}'.")
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Video file '{file_path.name}' contains no valid or decodable video stream.",
                )

            # Extract video technical properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or first_frame.shape[1]
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or first_frame.shape[0]
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if fps <= 0 or fps > 240:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Video file '{file_path.name}' has invalid FPS metadata.",
                )

            raw_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if raw_frame_count <= 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Video file '{file_path.name}' has invalid frame-count metadata.",
                )

            duration = round(raw_frame_count / fps, 2)
            if duration > settings.MAX_VIDEO_DURATION_SECONDS:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Video duration exceeds the maximum allowed limit of {settings.MAX_VIDEO_DURATION_SECONDS} seconds.",
                )
            normalized_format = "mp4"
            rotation = self._video_rotation(file_path)

            # Frame extraction: setup output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            extracted_frames: List[ExtractedFrame] = []

            interval_seconds = max(0.5, frame_interval if frame_interval and frame_interval > 0 else 1.0)
            step_frames = max(1, int(round(fps * interval_seconds)))
            max_samples = settings.MAX_FRAMES_PER_VIDEO

            current_frame_idx = 0
            saved_count = 0

            while cap.isOpened() and saved_count < max_samples:
                success, frame = cap.read()
                if not success or frame is None:
                    break

                if current_frame_idx % step_frames == 0:
                    frame = self.rotate_frame(frame, rotation)
                    saved_count += 1
                    frame_filename = f"{media_id}_frame_{current_frame_idx:04d}.jpg"
                    frame_path = output_dir / frame_filename

                    # Save RGB frame JPEG
                    cv2.imwrite(str(frame_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

                    timestamp = round(current_frame_idx / fps, 2)
                    relative_url = file_manager.get_relative_url(frame_path)

                    extracted_frames.append(
                        ExtractedFrame(
                            frame_number=current_frame_idx,
                            timestamp_seconds=timestamp,
                            path=frame_path,
                            relative_url=relative_url,
                        )
                    )

                current_frame_idx += 1

            if not extracted_frames:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Video file '{file_path.name}' contains no sampled frames.",
                )

            return NormalizedVideoResult(
                width=width,
                height=height,
                duration=duration,
                frame_count=raw_frame_count,
                fps=round(fps, 2),
                normalized_format=normalized_format,
                rotation=rotation,
                extracted_frames=extracted_frames,
            )
        finally:
            cap.release()

    @staticmethod
    def _video_rotation(file_path: Path) -> int:
        """Read common phone-video rotation metadata when the decoder ignores it."""
        ffprobe = shutil.which("ffprobe")
        if not ffprobe:
            return 0
        result = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream_tags=rotate:side_data_list", "-of", "default=nw=1", str(file_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        match = re.search(r"(?:rotate|rotation)\s*[=:]\s*(-?\d+(?:\.\d+)?)", result.stdout + result.stderr, re.IGNORECASE)
        if not match:
            return 0
        return int(round(float(match.group(1)))) % 360

    @staticmethod
    def rotate_frame(frame, rotation: int):
        if rotation == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if rotation == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        if rotation == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame


media_validator = MediaValidationNormalizationService()
