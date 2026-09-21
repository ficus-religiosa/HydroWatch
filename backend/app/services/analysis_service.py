import secrets
import asyncio
import io
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import UploadFile, HTTPException, status
from starlette.datastructures import Headers

from app.core.config import settings
from app.schemas.analysis import (
    AnalysisResponse,
    FrameResult,
    AnalysisAccepted,
    LocationInfo,
    MediaMetadata,
)
from app.services.geocoding_service import geocoding_service
from app.services.media_service import media_service
from app.services.media_validator import media_validator
from app.services.analysis_pipeline import build_analysis_pipeline
from app.services.inference.factory import get_inference_engine
from app.repository.analysis_repository import AnalysisRepository, FileAnalysisRepository
from app.job.runner import JobRunner, job_runner
from app.storage.file_manager import file_manager
from app.storage.file_manager import request_base_url
from app.utils.logger import logger


class AnalysisService:
    """Coordinates media upload intake, validation, storage, and analysis orchestration."""

    def __init__(self, repository: AnalysisRepository | None = None, runner: JobRunner | None = None):
        self.repository = repository or FileAnalysisRepository(settings.UPLOAD_DIR)
        self.runner = runner or job_runner
        self.pipeline = build_analysis_pipeline(get_inference_engine())

    def get_analysis(self, analysis_id: str) -> Optional[AnalysisResponse]:
        """Retrieve an existing analysis by ID from memory or disk cache."""
        return self.repository.get(analysis_id)

    def _persist_status(self, analysis_id: str, record: AnalysisResponse) -> None:
        self.repository.save(record)

    async def enqueue_analysis(
        self,
        files: List[UploadFile],
        location: Optional[str] = None,
        frame_interval: Optional[float] = None,
        confidence_threshold: Optional[float] = None,
        public_base_url: Optional[str] = None,
    ) -> tuple[AnalysisAccepted, asyncio.Task]:
        """Create a queued record and schedule local background processing."""
        analysis_id = self._new_analysis_id()
        record = AnalysisResponse(
            analysis_id=analysis_id,
            status="queued",
            progress=0,
            message="Analysis queued",
            created_at=datetime.now(timezone.utc),
        )
        self.repository.save(record)
        queued_files: List[UploadFile] = []
        for upload_file in files:
            content = await upload_file.read()
            queued_files.append(
                UploadFile(
                    file=io.BytesIO(content),
                    filename=upload_file.filename,
                    headers=Headers({"content-type": upload_file.content_type or ""}),
                )
            )
        base_url = public_base_url or settings.PUBLIC_BASE_URL
        task = self.runner.submit(
            self.process_analysis(
                analysis_id=analysis_id,
                files=queued_files,
                location=location,
                frame_interval=frame_interval,
                confidence_threshold=confidence_threshold,
                public_base_url=base_url,
            )
        )
        return AnalysisAccepted(analysis_id=analysis_id), task

    @staticmethod
    def _new_analysis_id() -> str:
        """Return an analysis_<ULID>-shaped identifier without a new dependency."""
        alphabet = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
        value = (int(time.time() * 1000) << 80) | secrets.randbits(80)
        encoded = []
        for _ in range(26):
            encoded.append(alphabet[value & 31])
            value >>= 5
        return f"analysis_{''.join(reversed(encoded))}"

    async def process_analysis(
        self,
        analysis_id: str,
        files: List[UploadFile],
        location: Optional[str] = None,
        frame_interval: Optional[float] = None,
        confidence_threshold: Optional[float] = None,
        public_base_url: Optional[str] = None,
    ) -> None:
        """Run the existing media/inference pipeline as a local background job."""
        record = self.repository.get(analysis_id)
        if not record:
            return
        record.status = "processing"
        record.message = "Preparing uploaded media"
        self._persist_status(analysis_id, record)
        try:
            token = request_base_url.set(public_base_url)
            result = await self._process_analysis(
                analysis_id=analysis_id,
                files=files,
                location=location,
                frame_interval=frame_interval,
                confidence_threshold=confidence_threshold,
            )
            request_base_url.reset(token)
        except HTTPException as exc:
            record.status = "failed"
            record.message = str(exc.detail)
            record.completed_at = datetime.now(timezone.utc)
            self._persist_status(analysis_id, record)
        except Exception:
            logger.exception("Analysis %s failed", analysis_id)
            record.status = "failed"
            record.message = "Analysis failed while processing the submitted media."
            record.completed_at = datetime.now(timezone.utc)
            self._persist_status(analysis_id, record)

    async def _process_analysis(
        self,
        analysis_id: str,
        files: List[UploadFile],
        location: Optional[str] = None,
        frame_interval: Optional[float] = None,
        confidence_threshold: Optional[float] = None,
        public_base_url: Optional[str] = None,
    ) -> AnalysisResponse:
        """
        Process uploaded media files, validate constraints, store them securely,
        and generate a complete analysis response conforming to BACKEND_HANDOFF.md.
        """
        if not files or len(files) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least one image or video file must be provided in the 'files' field.",
            )

        if len(files) > settings.MAX_TOTAL_FILES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Exceeded maximum allowed files per request ({settings.MAX_TOTAL_FILES}).",
            )

        upload_dir = file_manager.get_analysis_upload_dir(analysis_id)

        # Geocode location if provided
        location_info: Optional[LocationInfo] = None
        warning = None
        if location and location.strip():
            geo_res = await geocoding_service.geocode(location)
            if geo_res:
                if geo_res.get("location"):
                    location_info = LocationInfo(**geo_res["location"])
                warning = geo_res.get("warning")

        media_list: List[MediaMetadata] = []
        frames_list: List[FrameResult] = []

        threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else settings.CONFIDENCE_THRESHOLD_DEFAULT
        )
        if not 0.0 <= threshold <= 1.0:
            raise HTTPException(status_code=400, detail="confidence_threshold must be between 0 and 1.")

        interval = (
            frame_interval
            if frame_interval is not None
            else settings.FRAME_INTERVAL_DEFAULT_SECONDS
        )
        if not settings.FRAME_INTERVAL_MIN_SECONDS <= interval <= settings.FRAME_INTERVAL_MAX_SECONDS:
            raise HTTPException(
                status_code=400,
                detail=(
                    "frame_interval must be between "
                    f"{settings.FRAME_INTERVAL_MIN_SECONDS} and "
                    f"{settings.FRAME_INTERVAL_MAX_SECONDS} seconds."
                ),
            )

        for idx, upload_file in enumerate(files):
            # 1. Validate file format and content type using dedicated media_validator
            kind, ext = media_validator.validate_file_metadata(upload_file)
            safe_name = media_service.sanitize_filename(upload_file.filename or f"media_{idx+1}{ext}")
            media_id = f"media_{idx + 1:03d}"

            # 2. Save file safely
            saved_path = upload_dir / f"{media_id}_{safe_name}"
            file_manager.save_upload_file(upload_file, saved_path)

            # 3. Validate file size (non-empty & within limits)
            media_validator.validate_file_size(saved_path)

            frames_dir = file_manager.get_analysis_frames_dir(analysis_id) / media_id

            if kind == "image":
                # 4. Validate integrity & normalize image into RGB JPEG, extracting width/height/format
                norm_img = media_validator.validate_and_normalize_image(
                    file_path=saved_path, output_dir=frames_dir, media_id=media_id
                )

                media_list.append(
                    MediaMetadata(
                        id=media_id,
                        name=safe_name,
                        kind="image",
                        original_format=norm_img.format,
                        model_format="jpeg",
                        normalized_format="jpeg",
                        width=norm_img.width,
                        height=norm_img.height,
                        url=norm_img.relative_url,
                    )
                )

                frames_list.extend(
                    await self.pipeline.run_analysis(
                        analysis_id=analysis_id,
                        media_id=media_id,
                        media_name=safe_name,
                        kind="image",
                        media_meta=media_list[-1],
                        norm_result=norm_img,
                        confidence_threshold=threshold,
                        detection_index_offset=idx * 10,
                    )
                )

            else:  # video
                # 4. Validate integrity with OpenCV, extract metadata (duration, frame_count, width, height, fps, normalized_format) & extract normalized frames
                norm_vid = media_validator.validate_and_normalize_video(
                    file_path=saved_path, output_dir=frames_dir, media_id=media_id, frame_interval=interval
                )

                media_list.append(
                    MediaMetadata(
                        id=media_id,
                        name=safe_name,
                        kind="video",
                        original_format=ext.lstrip("."),
                        model_format=norm_vid.normalized_format,
                        normalized_format=norm_vid.normalized_format,
                        duration_seconds=norm_vid.duration,
                        duration=norm_vid.duration,
                        frame_count=norm_vid.frame_count,
                        fps=norm_vid.fps,
                        width=norm_vid.width,
                        height=norm_vid.height,
                        url=file_manager.get_relative_url(saved_path),
                    )
                )

                frames_list.extend(
                    await self.pipeline.run_analysis(
                        analysis_id=analysis_id,
                        media_id=media_id,
                        media_name=safe_name,
                        kind="video",
                        media_meta=media_list[-1],
                        norm_result=norm_vid,
                        confidence_threshold=threshold,
                        detection_index_offset=idx * 10,
                        on_frame=lambda current, total, media_index=idx: self._update_frame_progress(
                            analysis_id, media_index, len(files), current, total
                        ),
                    )
                )

        # 5. Compute summary statistics
        classes, size_stats, avg_confidence = self.pipeline.aggregate_statistics(
            frames_list
        )
        total_detections = sum(len(frame.detections) for frame in frames_list)
        overall_density = frames_list[0].density if frames_list else {"value": 0.0, "unit": "items/frame"}

        # 6. Build response
        response = AnalysisResponse(
            analysis_id=analysis_id,
            status="completed",
            progress=100,
            message="Analysis completed",
            created_at=self.repository.get(analysis_id).created_at,
            completed_at=datetime.now(timezone.utc),
            location=location_info,
            warning=warning,
            model_info=self.pipeline.inference_engine.model_info,
            media=media_list,
            total_detections=total_detections,
            average_confidence=avg_confidence,
            classes=classes,
            size_stats=size_stats,
            density=overall_density,
            frames=frames_list,
        )

        # 7. Persist to memory and disk
        self.repository.save(response)

        logger.info(
            f"Created analysis {analysis_id} with {len(files)} media files, "
            f"{len(frames_list)} frames, {total_detections} detections."
        )

        return response

    def _update_frame_progress(self, analysis_id: str, media_index: int, media_total: int, current: int, total: int) -> None:
        record = self.repository.get(analysis_id)
        if not record:
            return
        record.progress = min(95, int(((media_index + (current / max(total, 1))) / media_total) * 90) + 5)
        record.message = f"Processing frame {current} of {total}"
        self._persist_status(analysis_id, record)


analysis_service = AnalysisService()
