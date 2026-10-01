"""Place name -> coordinates, via OpenStreetMap Nominatim, cached in MongoDB.
Nominatim's usage policy allows ~1 request per second with an identifying User-Agent; the cache
means each distinct place name is looked up once."""
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from .config import settings
from .db import db

_last_call = [0.0]


def geocode(query):
    """Returns {"label", "latitude", "longitude"} or None."""
    q = (query or "").strip()
    if not q or settings.GEOCODER == "off":
        return None
    key = q.lower()
    cached = db().geocode_cache.find_one({"_id": key})
    if cached:
        return cached.get("result")
    wait = 1.1 - (time.time() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.time()
    url = "https://nominatim.openstreetmap.org/search?" + urllib.parse.urlencode(
        {"format": "jsonv2", "limit": 1, "q": q})
    try:
        req = urllib.request.Request(url, headers={"User-Agent": settings.GEOCODER_USER_AGENT,
                                                   "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=8) as r:
            hits = json.loads(r.read().decode("utf-8"))
    except Exception:
        return None   # offline or rate-limited: don't cache, try again next time
    result = None
    if hits:
        h = hits[0]
        result = {"label": h.get("display_name", q), "latitude": round(float(h["lat"]), 6),
                  "longitude": round(float(h["lon"]), 6)}
    db().geocode_cache.replace_one({"_id": key}, {"_id": key, "result": result,
                                                  "created_at": datetime.now(timezone.utc)}, upsert=True)
    return result
