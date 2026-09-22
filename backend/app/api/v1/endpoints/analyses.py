from typing import List, Optional
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse

from app.schemas.analysis import AnalysisAccepted, AnalysisResponse, FrameResult
from app.services.analysis_service import analysis_service
from app.services.annotation_service import annotation_service
from app.services.hotspot_service import hotspot_service
from app.services.impact_service import impact_service
from app.services.severity_service import severity_service
from app.schemas.hotspot import HotspotItem
from app.core.auth import get_current_client

router = APIRouter(prefix="/analyses", tags=["Analyses"], dependencies=[Depends(get_current_client)])


@router.post(
    "",
    response_model=AnalysisAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload media and run debris detection analysis",
    description="Accepts multiple images (JPEG, PNG, WebP, BMP) or videos (MP4, WebM, MOV, AVI), validates inputs, safely stores them, and returns detection results.",
)
async def create_analysis(
    request: Request,
    files: List[UploadFile] = File(..., description="One or more image or video files to analyze"),
    location: Optional[str] = Form(None, description="Optional user location string (e.g., 'India, Gujarat')"),
    frame_interval: Optional[float] = Form(1.0, description="Video sampling interval in seconds"),
    confidence_threshold: Optional[float] = Form(0.5, description="Confidence threshold (0.0 to 1.0)"),
) -> AnalysisAccepted:
    accepted, _ = await analysis_service.enqueue_analysis(
        files=files,
        location=location,
        frame_interval=frame_interval,
        confidence_threshold=confidence_threshold,
        public_base_url=str(request.base_url),
    )
    return accepted


@router.get(
    "/{analysis_id}",
    response_model=AnalysisResponse,
    summary="Get analysis results by ID",
    description="Retrieve the complete analysis result, summary metrics, and frame review data.",
)
async def get_analysis(analysis_id: str) -> AnalysisResponse:
    analysis = analysis_service.get_analysis(analysis_id)
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis with ID '{analysis_id}' not found.",
        )
    return analysis


@router.get(
    "/{analysis_id}/frames",
    response_model=List[FrameResult],
    summary="Get all reviewable frames for an analysis",
    description="Retrieve the list of frames or photos with detection bounding boxes.",
)
async def get_analysis_frames(
    analysis_id: str,
    media_id: Optional[str] = Query(None),
    label: Optional[str] = Query(None),
) -> List[FrameResult]:
    analysis = analysis_service.get_analysis(analysis_id)
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis with ID '{analysis_id}' not found.",
        )
    if media_id and not any(media.id == media_id for media in analysis.media):
        raise HTTPException(status_code=404, detail="Media not found in analysis.")
    return [
        frame for frame in analysis.frames
        if (not media_id or frame.media_id == media_id)
        and (not label or any(detection.label == label for detection in frame.detections))
    ]


@router.get(
    "/{analysis_id}/frames/{frame_id}",
    response_model=FrameResult,
    summary="Get details and detections for a specific frame",
)
async def get_analysis_frame(analysis_id: str, frame_id: str) -> FrameResult:
    analysis = analysis_service.get_analysis(analysis_id)
    if not analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis with ID '{analysis_id}' not found.",
        )
    for frame in analysis.frames:
        if frame.id == frame_id:
            return frame
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Frame '{frame_id}' not found in analysis '{analysis_id}'.",
    )


@router.get("/{analysis_id}/frames/{frame_id}/annotated")
async def get_annotated_frame(analysis_id: str, frame_id: str):
    path = annotation_service.annotate_frame(analysis_id, frame_id)
    return FileResponse(path, media_type="image/jpeg", filename=f"{frame_id}-annotated.jpg")


@router.get("/{analysis_id}/media/{media_id}/annotated")
async def get_annotated_media(analysis_id: str, media_id: str):
    analysis = analysis_service.get_analysis(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    media = next((item for item in analysis.media if item.id == media_id), None)
    if media is None:
        raise HTTPException(status_code=404, detail="Media not found in analysis.")
    if media.kind == "video":
        state, path, error = annotation_service.request_video_annotation(analysis_id, media_id)
        if state == "processing":
            return JSONResponse(status_code=202, content={"status": "processing", "progress": 0})
        if state == "failed":
            raise HTTPException(status_code=503, detail={"code": "annotation_failed", "message": error or "Annotated video generation failed."})
        return FileResponse(path, media_type="video/mp4", filename=f"{media_id}-annotated.mp4")
    frame = next(item for item in analysis.frames if item.media_id == media_id)
    path = annotation_service.annotate_frame(analysis_id, frame.id)
    return FileResponse(path, media_type="image/jpeg", filename=f"{media_id}-annotated.jpg")


@router.get("/{analysis_id}/report")
async def get_report(analysis_id: str, media_id: Optional[str] = Query(None)):
    analysis = analysis_service.get_analysis(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    frames = [frame for frame in analysis.frames if not media_id or frame.media_id == media_id]
    if media_id and not any(media.id == media_id for media in analysis.media):
        raise HTTPException(status_code=404, detail="Media not found in analysis.")
    detection_count = sum(len(frame.detections) for frame in frames)
    report_analysis = analysis.model_copy(update={"frames": frames, "total_detections": detection_count})
    return {
        "analysis_id": analysis_id,
        "timestamp": analysis.completed_at or analysis.created_at,
        "media": [media for media in analysis.media if not media_id or media.id == media_id],
        "location": analysis.location,
        "total_detections": detection_count,
        "categories": analysis.classes,
        "average_confidence": analysis.average_confidence,
        "size_stats": analysis.size_stats,
        "density": analysis.density,
        "model_info": analysis.model_info,
        "severity": severity_service.evaluate(report_analysis),
        "impact": impact_service.evaluate(report_analysis),
        "hotspot": hotspot_service.for_analysis(analysis_id),
    }


@router.get("/{analysis_id}/hotspots", response_model=List[HotspotItem])
async def get_analysis_hotspots(analysis_id: str):
    if not analysis_service.get_analysis(analysis_id):
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return hotspot_service.for_analysis(analysis_id)
