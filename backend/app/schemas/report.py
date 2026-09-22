from typing import List, Optional
from pydantic import BaseModel, Field


class ReportResponse(BaseModel):
    analysis_id: str
    survey_summary: str
    total_debris: int
    categories: List[str] = Field(default_factory=list)
    severity: str = "Moderate"
    impact: str
    cleanup_priority: str = "Priority 2"
    timestamp: str
