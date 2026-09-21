from app.core.config import settings


class SeverityService:
    """Prototype severity; spatial spread means bbox-center dispersion within each frame."""
    def evaluate(self, analysis):
        count = analysis.total_detections
        if count >= settings.PROTOTYPE_SEVERITY_CRITICAL_COUNT:
            category = "Critical"
        elif count >= settings.PROTOTYPE_SEVERITY_HIGH_COUNT:
            category = "High"
        elif count >= settings.PROTOTYPE_SEVERITY_MODERATE_COUNT:
            category = "Moderate"
        else:
            category = "Low"
        return {
            "score": min(100, count * settings.PROTOTYPE_SEVERITY_SCORE_WEIGHT_COUNT),
            "category": category,
            "supporting_stats": {"total_detections": count, "frames": len(analysis.frames)},
            "methodology": "prototype",
        }


severity_service = SeverityService()