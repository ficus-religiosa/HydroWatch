from datetime import datetime, timezone
from typing import List, Optional
from pydantic import BaseModel, Field, model_validator


class BoundingBox(BaseModel):
    x: float = Field(..., description="X coordinate (normalized 0-1 or percentage 0-100)")
    y: float = Field(..., description="Y coordinate (normalized 0-1 or percentage 0-100)")
    width: float = Field(..., description="Width (normalized 0-1 or percentage 0-100)")
    height: float = Field(..., description="Height (normalized 0-1 or percentage 0-100)")


class DetectionItem(BaseModel):
    id: str = Field(..., description="Unique detection identifier")
    label: str = Field(..., description="Debris class label, e.g., Plastic Bottle")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    size: str = Field(default="medium", description="Object size: small, medium, large")
    bbox: Optional[BoundingBox] = Field(default=None, description="Bounding box details")
    # Coordinates for direct consumption by frontend (e.g., App.jsx style left/top %)
    x: float = Field(default=0.0)
    y: float = Field(default=0.0)
    width: float = Field(default=0.0)
    height: float = Field(default=0.0)

    @model_validator(mode="after")
    def populate_bbox_and_coords(self):
        if self.bbox is None:
            self.bbox = BoundingBox(x=self.x, y=self.y, width=self.width, height=self.height)
        else:
            if self.x == 0.0 and self.y == 0.0 and self.width == 0.0 and self.height == 0.0:
                self.x = self.bbox.x
                self.y = self.bbox.y
                self.width = self.bbox.width
                self.height = self.bbox.height
        return self


class DensityInfo(BaseModel):
    value: float = Field(default=0.0)
    unit: str = Field(default="items/frame")


class FrameResult(BaseModel):
    id: str = Field(..., description="Unique frame identifier")
    media_id: Optional[str] = Field(default=None)
    frame_number: int = Field(default=1)
    timestamp_seconds: float = Field(default=0.0)
    label: str = Field(..., description="Frame label, e.g., underwater.mp4 / frame 1")
    kind: str = Field(default="image", description="'image' or 'video'")
    density: DensityInfo = Field(default_factory=DensityInfo)
    preview_url: Optional[str] = Field(default=None)
    annotated_preview_url: Optional[str] = Field(default=None)
    mediaUrl: Optional[str] = Field(default=None, description="Frontend alias for preview URL")
    detections: List[DetectionItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def sync_media_urls(self):
        if not self.mediaUrl and self.preview_url:
            self.mediaUrl = self.preview_url
        elif not self.preview_url and self.mediaUrl:
            self.preview_url = self.mediaUrl
        return self


class MediaMetadata(BaseModel):
    id: Optional[str] = None
    name: str
    kind: str = "image"
    original_format: Optional[str] = None
    model_format: Optional[str] = None
    normalized_format: Optional[str] = None
    duration_seconds: Optional[float] = None
    duration: Optional[float] = None
    frame_count: Optional[int] = None
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    url: Optional[str] = None

    @model_validator(mode="after")
    def sync_media_aliases(self):
        if not self.normalized_format and self.model_format:
            self.normalized_format = self.model_format
        elif not self.model_format and self.normalized_format:
            self.model_format = self.normalized_format

        if self.duration is None and self.duration_seconds is not None:
            self.duration = self.duration_seconds
        elif self.duration_seconds is None and self.duration is not None:
            self.duration_seconds = self.duration

        return self


class LocationInfo(BaseModel):
    query: Optional[str] = None
    label: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class ClassStat(BaseModel):
    label: str
    count: int


class SizeStats(BaseModel):
    small: int = 0
    medium: int = 0
    large: int = 0


class AnalysisResponse(BaseModel):
    analysis_id: str
    status: str = Field(default="completed", description="queued, processing, completed, failed")
    progress: int = Field(default=0, ge=0, le=100)
    message: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    warning: Optional[dict] = None
    location: Optional[LocationInfo] = None
    media: List[MediaMetadata] = Field(default_factory=list)
    # Snake_case (BACKEND_HANDOFF.md)
    total_detections: int = 0
    average_confidence: float = 0.0
    classes: List[ClassStat] = Field(default_factory=list)
    size_stats: SizeStats = Field(default_factory=SizeStats)
    density: DensityInfo = Field(default_factory=DensityInfo)
    frames: List[FrameResult] = Field(default_factory=list)
    # CamelCase aliases for direct frontend consumption
    totalDetections: Optional[int] = None
    averageConfidence: Optional[float] = None
    sizeStats: Optional[SizeStats] = None
    model_info: dict = Field(default_factory=lambda: {"name": "mock", "is_mock": True})

    @model_validator(mode="after")
    def sync_camel_case(self):
        if self.totalDetections is None:
            self.totalDetections = self.total_detections
        if self.averageConfidence is None:
            self.averageConfidence = self.average_confidence
        if self.sizeStats is None:
            self.sizeStats = self.size_stats
        return self


class AnalysisAccepted(BaseModel):
    analysis_id: str
    status: str = "queued"
    progress: int = 0
    message: str = "Analysis queued"
