from app.core.config import settings


class ImpactService:
    def evaluate(self, analysis):
        return {
            "status": "prototype",
            "summary": settings.PROTOTYPE_IMPACT_SUMMARY,
            "methodology": "prototype",
            "categories": [item.label for item in analysis.classes],
        }


impact_service = ImpactService()