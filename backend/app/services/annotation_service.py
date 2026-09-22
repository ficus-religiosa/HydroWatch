import shutil
import subprocess
import os
import cv2
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image, ImageDraw
from fastapi import HTTPException, status

from app.core.config import settings
from app.services.analysis_service import analysis_service
from app.services.media_validator import MediaValidationNormalizationService


class AnnotationService:
    def __init__(self):
        self._video_jobs = {}
        self._video_locks = {}
        self._state_lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="hydrowatch-annotate")

    def annotate_frame(self, analysis_id: str, frame_id: str) -> Path:
        analysis = analysis_service.get_analysis(analysis_id)
        frame = next((item for item in analysis.frames if item.id == frame_id), None) if analysis else None
        if frame is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Frame not found.")
        source = self._source_path(frame.preview_url)
        if not source or not source.exists():
            raise HTTPException(status_code=404, detail="Frame preview is unavailable.")
        output = settings.ANNOTATED_DIR / analysis_id / f"{frame_id}.jpg"
        output.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source).convert("RGB") as image:
            draw = ImageDraw.Draw(image)
            for detection in frame.detections:
                box = detection.bbox
                xy = (box.x * image.width, box.y * image.height, (box.x + box.width) * image.width, (box.y + box.height) * image.height)
                draw.rectangle(xy, outline="red", width=3)
                draw.text((xy[0], max(0, xy[1] - 14)), f"{detection.label} {detection.confidence:.2f}", fill="red")
            image.save(output, "JPEG", quality=95)
        return output

    def request_video_annotation(self, analysis_id: str, media_id: str):
        key = (analysis_id, media_id)
        output = settings.ANNOTATED_DIR / analysis_id / f"{media_id}-annotated.mp4"
        with self._state_lock:
            if output.exists():
                return "ready", output, None
            existing = self._video_jobs.get(key)
            if existing:
                return existing["status"], None, existing.get("error")
            lock = self._video_locks.setdefault(key, threading.Lock())
            self._video_jobs[key] = {"status": "processing", "progress": 0, "error": None}
            self._executor.submit(self._annotate_video_job, analysis_id, media_id, lock)
            return "processing", None, None

    def _annotate_video_job(self, analysis_id: str, media_id: str, lock: threading.Lock) -> None:
        key = (analysis_id, media_id)
        with lock:
            try:
                self._encode_video(analysis_id, media_id)
                with self._state_lock:
                    self._video_jobs[key] = {"status": "ready", "progress": 100, "error": None}
            except HTTPException as exc:
                with self._state_lock:
                    self._video_jobs[key] = {"status": "failed", "progress": 0, "error": str(exc.detail)}
            except Exception:
                with self._state_lock:
                    self._video_jobs[key] = {"status": "failed", "progress": 0, "error": "Annotated video generation failed."}

    def _encode_video(self, analysis_id: str, media_id: str) -> Path:
        analysis = analysis_service.get_analysis(analysis_id)
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found.")
        media = next((item for item in analysis.media if item.id == media_id), None)
        if not media:
            raise HTTPException(status_code=404, detail="Media not found in analysis.")
        source = self._source_path(media.url)
        if not source or not source.exists():
            raise HTTPException(status_code=404, detail="Source video is unavailable.")

        output = settings.ANNOTATED_DIR / analysis_id / f"{media_id}-annotated.mp4"
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except (ImportError, RuntimeError):
            raise HTTPException(
                status_code=503,
                detail={"code": "ffmpeg_unavailable", "message": "Annotated video encoding is unavailable."},
            )

        frame_by_number = {frame.frame_number: frame for frame in analysis.frames if frame.media_id == media_id}
        temp_dir = settings.ANNOTATED_DIR / analysis_id / f"{media_id}-tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_output = temp_dir / "annotated.mp4"
        try:
            cap = cv2.VideoCapture(str(source))
            if not cap.isOpened():
                raise HTTPException(status_code=400, detail="Source video could not be opened.")
            fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            rotation = MediaValidationNormalizationService._video_rotation(source)
            if rotation in {90, 270}:
                width, height = height, width
            process = subprocess.Popen(
                [ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temp_output)],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            last_frame = None
            frame_number = 0
            while True:
                success, image = cap.read()
                if not success:
                    break
                image = MediaValidationNormalizationService.rotate_frame(image, rotation)
                if frame_number in frame_by_number:
                    last_frame = frame_by_number[frame_number]
                if last_frame:
                    self._draw_detections(image, last_frame)
                process.stdin.write(image.tobytes())
                frame_number += 1
            cap.release()
            process.stdin.close()
            try:
                return_code = process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                raise HTTPException(status_code=503, detail="Annotated video encoding timed out.")
            if return_code != 0:
                raise HTTPException(status_code=503, detail="Annotated video encoding failed.")
            output.parent.mkdir(parents=True, exist_ok=True)
            os.replace(temp_output, output)
            return output
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    @staticmethod
    def _draw_detections(image, frame) -> None:
        height, width = image.shape[:2]
        for detection in frame.detections:
            box = detection.bbox
            x1 = int(box.x * width)
            y1 = int(box.y * height)
            x2 = int((box.x + box.width) * width)
            y2 = int((box.y + box.height) * height)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(image, f"{detection.label} {detection.confidence:.2f}", (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    @staticmethod
    def _source_path(url: str | None) -> Path | None:
        if not url:
            return None
        marker = "/static/media/"
        if marker not in url:
            return None
        relative = url.split(marker, 1)[1]
        candidate = (settings.STORAGE_BASE_DIR / relative).resolve()
        if settings.STORAGE_BASE_DIR.resolve() not in candidate.parents:
            return None
        return candidate


annotation_service = AnnotationService()