from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query

from app.schemas.hotspot import HotspotItem
from app.services.analysis_service import analysis_service
from app.services.hotspot_service import hotspot_service
from app.core.auth import get_current_client

router = APIRouter(prefix="/hotspots", tags=["Hotspots"], dependencies=[Depends(get_current_client)])


@router.get("", response_model=List[HotspotItem])
async def get_hotspots(analysis_id: Optional[str] = Query(None)):
    if not analysis_id:
        return []
    if not analysis_service.get_analysis(analysis_id):
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return hotspot_service.for_analysis(analysis_id)