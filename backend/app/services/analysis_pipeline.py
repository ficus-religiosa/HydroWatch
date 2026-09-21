"""Inference pipeline orchestration independent of any concrete model engine."""

import asyncio
from typing import Callable, Dict, List, Optional

from app.schemas.analysis import BoundingBox, ClassStat, DensityInfo, DetectionItem, FrameResult, MediaMetadata, SizeStats
from app.services.inference.base import BaseInferenceEngine
from app.services.media_validator import NormalizedImageResult, NormalizedVideoResult


def build_detection_items(raw_detections: List[Dict], media_id: str, frame_number: int, confidence_threshold: float, index_offset: int = 0) -> List[DetectionItem]:
    items = []
    detection_index = 0
    for raw in raw_detections:
        if raw["confidence"] < confidence_threshold:
            continue
        detection_index += 1
        bbox = raw.get("bbox") or {key: raw.get(key, 0.0) / 100.0 for key in ("x", "y", "width", "height")}
        media_short = media_id.replace("media_", "m")
        items.append(DetectionItem(
            id=f"det_{media_short}_{frame_number:04d}_{detection_index + index_offset:02d}",
            label=raw["label"],
            confidence=raw["confidence"],
            size=raw.get("size", "medium"),
            bbox=BoundingBox(**bbox),
            x=bbox["x"] * 100,
            y=bbox["y"] * 100,
            width=bbox["width"] * 100,
            height=bbox["height"] * 100,
        ))
    return items


class AnalysisPipeline:
    def __init__(self, inference_engine: BaseInferenceEngine):
        self.inference_engine = inference_engine

    async def run_analysis(self, *, analysis_id: str, media_id: str, media_name: str, kind: str, media_meta: MediaMetadata, norm_result, confidence_threshold: float, on_frame: Optional[Callable[[int, int], None]] = None, detection_index_offset: int = 0) -> List[FrameResult]:
        if kind == "image":
            raw = self.inference_engine.detect_image(str(norm_result.normalized_path))
            detections = build_detection_items(raw, media_id, 1, confidence_threshold, detection_index_offset)
            return [FrameResult(id=f"{media_id}-frame-0001", media_id=media_id, frame_number=1, timestamp_seconds=0.0, label=media_name, kind="image", density=DensityInfo(value=len(detections), unit="items/frame"), preview_url=norm_result.relative_url, annotated_preview_url=norm_result.relative_url, detections=detections)]
        frames = []
        for extracted in norm_result.extracted_frames:
            raw = self.inference_engine.detect_image(str(extracted.path))
            detections = build_detection_items(raw, media_id, extracted.frame_number, confidence_threshold, detection_index_offset)
            frames.append(FrameResult(id=f"{media_id}-frame-{extracted.frame_number:04d}", media_id=media_id, frame_number=extracted.frame_number, timestamp_seconds=extracted.timestamp_seconds, label=f"{media_name} / frame {extracted.frame_number}", kind="video", density=DensityInfo(value=len(detections), unit="items/frame"), preview_url=extracted.relative_url, annotated_preview_url=extracted.relative_url, detections=detections))
            if on_frame:
                on_frame(len(frames), len(norm_result.extracted_frames))
                await asyncio.sleep(0)
        return frames

    def aggregate_statistics(self, frames: List[FrameResult]) -> tuple[List[ClassStat], SizeStats, float]:
        class_counts: Dict[str, int] = {}
        size_counts = {"small": 0, "medium": 0, "large": 0}
        confidences = []
        for frame in frames:
            for detection in frame.detections:
                class_counts[detection.label] = class_counts.get(detection.label, 0) + 1
                if detection.size in size_counts:
                    size_counts[detection.size] += 1
                confidences.append(detection.confidence)
        average = round(sum(confidences) / len(confidences) * 100, 1) if confidences else 0.0
        return [ClassStat(label=label, count=count) for label, count in class_counts.items()], SizeStats(**size_counts), average


def build_analysis_pipeline(inference_engine: BaseInferenceEngine) -> AnalysisPipeline:
    return AnalysisPipeline(inference_engine)