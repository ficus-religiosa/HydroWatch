from app.schemas.hotspot import HotspotItem
from app.services.analysis_service import analysis_service


class HotspotService:
    def for_analysis(self, analysis_id: str) -> list[HotspotItem]:
        analysis = analysis_service.get_analysis(analysis_id)
        if not analysis or not analysis.location:
            return []
        if analysis.location.latitude is None or analysis.location.longitude is None:
            return []
        detections = [detection for frame in analysis.frames for detection in frame.detections]
        categories = sorted({detection.label for detection in detections})
        return [HotspotItem(
            id=f"hotspot_{analysis_id}",
            name=analysis.location.label or analysis.location.query or "Survey location",
            latitude=analysis.location.latitude,
            longitude=analysis.location.longitude,
            count=len(detections),
            categories=categories,
        )]


hotspot_service = HotspotService()