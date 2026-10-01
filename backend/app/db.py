"""MongoDB access. The database and its collections are created automatically on first use.

Collections
  analyses        one document per upload request: status, location, media list, summary
  frames          one document per reviewed photo / video frame, with its detections
  locations       running totals per place - what the Hotspots map shows
  geocode_cache   place-name lookups, so the same name is only geocoded once
"""
from pymongo import ASCENDING, DESCENDING, MongoClient

from .config import settings

_client = None


def client():
    global _client
    if _client is None:
        if settings.MONGO_URI.startswith("mongomock://"):   # used by the automated tests only
            import mongomock
            _client = mongomock.MongoClient()
        else:
            _client = MongoClient(settings.MONGO_URI, serverSelectionTimeoutMS=3000)
    return _client


def db():
    return client()[settings.MONGO_DB]


def init_db():
    """Check the server is reachable and create indexes (this also creates the database)."""
    try:
        client().admin.command("ping")
    except Exception as e:
        raise RuntimeError(
            f"MongoDB is not reachable at {settings.MONGO_URI}. Start MongoDB (see backend/README.md) "
            f"or set MONGO_URI in backend/.env. Details: {e}") from e
    d = db()
    d.analyses.create_index([("created_at", DESCENDING)])
    d.analyses.create_index([("status", ASCENDING)])
    d.frames.create_index([("analysis_id", ASCENDING), ("order", ASCENDING)])
    d.locations.create_index([("updated_at", DESCENDING)])
    # anything left mid-way by a previous crash can never finish now
    d.analyses.update_many({"status": {"$in": ["queued", "processing"]}},
                           {"$set": {"status": "failed", "message": "The server restarted before this analysis finished."}})
    for a in d.analyses.find({"media.annotated_status": {"$in": ["queued", "rendering"]}}, {"media": 1}):
        media = [{**m, "annotated_status": "failed"} if m.get("annotated_status") in ("queued", "rendering") else m
                 for m in a["media"]]
        d.analyses.update_one({"_id": a["_id"]}, {"$set": {"media": media}})
