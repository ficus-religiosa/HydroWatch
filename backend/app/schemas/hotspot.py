from typing import List, Optional
from pydantic import BaseModel, Field


class HotspotItem(BaseModel):
    id: str
    name: str
    latitude: float
    longitude: float
    risk: str = Field(default="Moderate", description="Low, Moderate, High, Critical")
    count: int = 0
    categories: List[str] = Field(default_factory=list)
