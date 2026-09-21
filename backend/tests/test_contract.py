from pathlib import Path

import pytest
import cv2
from PIL import Image

from app.schemas.analysis import AnalysisResponse, BoundingBox, DetectionItem, DensityInfo, FrameResult
from app.services.inference.factory import get_inference_engine
from app.services.severity_service import severity_service
from app.services.media_validator import media_validator
from app.services.analysis_pipeline import build_detection_items
from app.repository.analysis_repository import FileAnalysisRepository
from app.storage.cleanup import cleanup_expired_storage
from app.core.config import settings
import threading


def test_detection_contract_uses_normalized_bbox_and_unit_confidence():
    engine = get_inference_engine()
    detections = engine.detect_image("synthetic-frame.jpg")
    assert detections
    assert all(0 <= item["confidence"] <= 1 for item in detections)
    assert all(0 <= item["bbox"][key] <= 1 for item in detections for key in ("x", "y", "width", "height"))
    assert detections == engine.detect_image("synthetic-frame.jpg")


def test_analysis_schema_contains_structured_density_and_model_info():
    response = AnalysisResponse(
        analysis_id="analysis_01JTEST",
        frames=[FrameResult(id="media_001-frame-0000", label="frame", density=DensityInfo(value=2, unit="items/frame"))],
    )
    assert response.density.unit == "items/frame"
    assert response.model_info["is_mock"] is True


def test_prototype_severity_is_labeled():
    response = AnalysisResponse(analysis_id="analysis_01JTEST", total_detections=6)
    result = severity_service.evaluate(response)
    assert result["methodology"] == "prototype"
    assert result["category"] == "Moderate"


def test_generated_video_extraction_uses_real_metadata(tmp_path):
    source = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
    for index in range(12):
        writer.write(__import__("numpy").full((24, 32, 3), index * 10, dtype="uint8"))
    writer.release()

    result = media_validator.validate_and_normalize_video(source, tmp_path / "frames", "media_001", 0.5)
    assert result.width == 32
    assert result.height == 24
    assert result.fps == 10.0
    assert result.frame_count == 12
    assert result.duration == 1.2
    assert [frame.frame_number for frame in result.extracted_frames] == [0, 5, 10]
    assert result.extracted_frames[-1].timestamp_seconds == 1.0


def test_pipeline_applies_confidence_threshold():
    raw = [
        {"id": "engine_01", "label": "A", "confidence": 0.2, "bbox": {"x": 0, "y": 0, "width": 0.1, "height": 0.1}},
        {"id": "engine_02", "label": "B", "confidence": 0.8, "bbox": {"x": 0, "y": 0, "width": 0.1, "height": 0.1}},
    ]
    detections = build_detection_items(raw, "media_001", 7, 0.5)
    assert [item.label for item in detections] == ["B"]
    assert detections[0].id == "det_m001_0007_01"


def test_repository_atomic_concurrent_writes(tmp_path):
    repository = FileAnalysisRepository(tmp_path)
    errors = []

    def save(progress):
        try:
            repository.save(AnalysisResponse(analysis_id="analysis_01JLOCK", status="processing", progress=progress))
        except Exception as error:
            errors.append(error)

    threads = [threading.Thread(target=save, args=(index,)) for index in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert errors == []
    assert repository.get("analysis_01JLOCK").status == "processing"


def test_frame_and_detection_id_patterns_are_unique():
    first = FrameResult(id="media_001-frame-0000", media_id="media_001", frame_number=0, label="one", detections=[DetectionItem(id="det_m001_0000_01", label="A", confidence=0.8, bbox=BoundingBox(x=0, y=0, width=0.1, height=0.1))])
    second = FrameResult(id="media_002-frame-0000", media_id="media_002", frame_number=0, label="two", detections=[DetectionItem(id="det_m002_0000_01", label="A", confidence=0.8, bbox=BoundingBox(x=0, y=0, width=0.1, height=0.1))])
    assert first.id != second.id
    assert first.detections[0].id != second.detections[0].id


@pytest.mark.parametrize("extension, fourcc", [("avi", "MJPG"), ("mp4", "mp4v"), ("webm", "VP80"), ("mov", "mp4v")])
def test_generated_video_containers_decode_or_report_clear_error(tmp_path, extension, fourcc):
    source = tmp_path / f"sample.{extension}"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*fourcc), 10.0, (32, 24))
    for index in range(6):
        writer.write(__import__("numpy").full((24, 32, 3), index * 10, dtype="uint8"))
    writer.release()
    if not source.exists() or source.stat().st_size == 0:
        pytest.skip(f"OpenCV could not create {extension} in this environment")
    try:
        result = media_validator.validate_and_normalize_video(source, tmp_path / f"frames-{extension}", "media_001", 0.5)
        assert result.frame_count == 6
        assert result.extracted_frames
    except Exception as error:
        detail = str(getattr(error, "detail", error)).lower()
        assert "corrupt" in detail or "container" in detail or "decodable" in detail or "codec" in detail