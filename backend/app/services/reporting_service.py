from typing import Any, Dict
from app.utils.logger import logger


class ReportingService:
    """Service to produce text or PDF briefing summaries."""

    def generate_text_report(self, analysis_data: Dict[str, Any], location: str = "") -> str:
        media_names = ", ".join([m.get("name", "") for m in analysis_data.get("media", [])]) or "No media selected"
        classes_str = ", ".join([f"{c['label']} ({c['count']})" for c in analysis_data.get("classes", [])])
        return (
            "HydroWatch Media Analysis Report\n\n"
            f"Media: {media_names}\n"
            f"Location: {location or 'Not supplied'}\n"
            f"Detected objects: {analysis_data.get('total_detections', 0)}\n"
            f"Frames or photos reviewed: {len(analysis_data.get('frames', []))}\n"
            f"Debris classes: {classes_str}\n\n"
            "This report describes the selected media only."
        )


reporting_service = ReportingService()
