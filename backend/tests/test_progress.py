import asyncio
import io
import time
from pathlib import Path

from fastapi import UploadFile
import cv2
import numpy as np
from PIL import Image

from app.repository.analysis_repository import FileAnalysisRepository
from app.services.analysis_service import AnalysisService
from app.services.analysis_pipeline import AnalysisPipeline
from app.services.inference.base import BaseInferenceEngine
from app.services.media_validator import media_validator


class SlowEngine(BaseInferenceEngine):
    @property
    def model_info(self):
        return {"name": "slow-test", "is_mock": True}

    def detect_image(self, image_path: str, confidence_threshold: float = 0.0):
        time.sleep(0.06)
        return [{"label": "Test", "confidence": 0.9, "bbox": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2}}]


def test_frame_progress_is_persisted_and_increases(tmp_path, monkeypatch):
    source = tmp_path / "slow.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
    for index in range(30):
        writer.write(np.full((24, 32, 3), index, dtype=np.uint8))
    writer.release()
    repository = FileAnalysisRepository(tmp_path / "uploads")
    service = AnalysisService(repository=repository)
    service.pipeline = AnalysisPipeline(SlowEngine())

    async def run():
        accepted, task = await service.enqueue_analysis([UploadFile(file=source.open("rb"), filename="slow.mp4", headers={"content-type": "video/mp4"})], frame_interval=0.5)
        observed = []
        while not task.done():
            record = repository.get(accepted.analysis_id)
            if record:
                observed.append((record.progress, record.message))
            await asyncio.sleep(0.01)
        await task
        return observed, repository.get(accepted.analysis_id)

    observed, result = asyncio.run(run())
    frame_messages = [item for item in observed if item[1].startswith("Processing frame")]
    print("PROGRESS_MESSAGES", frame_messages)
    assert len({message for _, message in frame_messages}) >= 2
    assert len({progress for progress, _ in frame_messages}) >= 2
    assert result.status == "completed"