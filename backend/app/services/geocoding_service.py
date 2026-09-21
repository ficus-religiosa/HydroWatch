import asyncio
import time
from typing import Optional, Dict
from app.utils.logger import logger
import httpx


class GeocodingService:
    def __init__(self):
        self._cache: Dict[str, Dict] = {}
        self._last_request_at = 0.0
        self._rate_lock = asyncio.Lock()

    async def _wait_for_rate_limit(self) -> None:
        async with self._rate_lock:
            wait_seconds = 1.0 - (time.monotonic() - self._last_request_at)
            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)
            self._last_request_at = time.monotonic()

    async def geocode(self, location_query: Optional[str]) -> Optional[Dict]:
        """
        Geocode a location query into latitude, longitude, and canonical label.
        Uses in-memory cache to avoid repeated external lookups.
        """
        if not location_query or not location_query.strip():
            return None

        normalized_query = location_query.strip().lower()
        if normalized_query in self._cache:
            return self._cache[normalized_query]

        warning = {"code": "geocoder_provider_error", "message": "The location provider could not resolve this request."}
        try:
            await self._wait_for_rate_limit()
            async with httpx.AsyncClient(timeout=5.0) as client:
                url = f"https://nominatim.openstreetmap.org/search?format=jsonv2&limit=1&q={location_query}"
                response = await client.get(
                    url,
                    headers={"User-Agent": "HydroWatch-Backend/0.1.0"},
                )
                if response.status_code == 200:
                    results = response.json()
                    if results and len(results) > 0:
                        first = results[0]
                        latitude = first.get("lat")
                        longitude = first.get("lon")
                        if latitude is None or longitude is None:
                            warning = {"code": "location_not_found", "message": "The location could not be resolved."}
                        else:
                            result = {
                                "location": {
                                    "query": location_query,
                                    "label": first.get("display_name", location_query),
                                    "latitude": float(latitude),
                                    "longitude": float(longitude),
                                },
                                "warning": None,
                            }
                            self._cache[normalized_query] = result
                            return result
                    warning = {"code": "location_not_found", "message": "The location could not be resolved."}
                elif response.status_code == 429:
                    warning = {"code": "geocoder_rate_limited", "message": "The location provider rate-limited this request."}
                else:
                    warning = {"code": "geocoder_provider_error", "message": "The location provider could not resolve this request."}
        except httpx.TimeoutException:
            warning = {"code": "geocoder_timeout", "message": "The location provider timed out."}
        except Exception as e:
            logger.warning(f"Geocoding lookup failed for '{location_query}': {e}")
            warning = {"code": "geocoder_provider_error", "message": "The location provider could not be reached."}

        result = {"location": None, "warning": warning}
        self._cache[normalized_query] = result
        return result


geocoding_service = GeocodingService()
