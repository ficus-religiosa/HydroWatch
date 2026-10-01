"""HTTP API under /api/v1 - the contract in frontend/BACKEND_HANDOFF.md, plus dashboard and health."""
import secrets
from collections import Counter
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from . import media as M
from . import pipeline as P
from .config import settings
from .db import db
from .detector import CLASS_INFO

router = APIRouter(prefix="/api/v1")

RISK_TEXT = {
    "rope_net": "rope and netting pose an entanglement risk to marine animals",
    "soft_debris": "bags, film and fabric are an ingestion and smothering risk",
    "bottle": "bottles persist for decades and break down into microplastics",
    "can_cup": "cans and cups add persistent metal and plastic litter",
    "pipe_tube": "pipes and tubes are bulky, long-lived debris",
    "rubber": "rubber items (tyres, gloves) are persistent and can leach chemicals",
    "wreckage": "wreckage can damage seabed habitat",
    "wood": "processed wood is usually lower-risk but indicates human dumping",
    "other_debris": "miscellaneous man-made debris is present",
}


def _iso(dt):
    return dt.replace(tzinfo=None).isoformat(timespec="seconds") + "Z" if dt else None


def _url(request, relpath):
    return f"{str(request.base_url).rstrip('/')}/media/{relpath}" if relpath else None


def _media_out(m):
    return {k: m.get(k) for k in ("id", "name", "kind", "original_format", "model_format", "duration_seconds",
                                  "frame_count", "width", "height", "annotated_status", "annotated_progress")}


def _frame_out(f, request):
    return {"id": f["id"], "media_id": f["media_id"], "frame_number": f["frame_number"],
            "timestamp_seconds": f["timestamp_seconds"], "label": f["label"], "kind": f["kind"],
            "width": f["width"], "height": f["height"], "density": f["density"], "debris_count": f["debris_count"],
            "preview_url": _url(request, f["preview"]), "annotated_preview_url": _url(request, f["annotated"]),
            "detections": f["detections"]}


def _analysis_out(a, request, with_frames=True):
    out = {"analysis_id": a["_id"], "status": a["status"], "progress": a.get("progress", 0),
           "message": a.get("message", ""), "created_at": _iso(a.get("created_at")),
           "completed_at": _iso(a.get("completed_at")),
           "location": a.get("location") or {"query": a["location_input"].get("query", ""),
                                             "label": a["location_input"].get("query", "")},
           "media": [_media_out(m) for m in a["media"]], "settings": a["settings"],
           "model_info": a.get("model_info")}
    if a["status"] == "completed":
        for k in ("total_detections", "debris_detections", "average_confidence", "classes", "size_stats",
                  "frames_reviewed", "density", "peak_debris_per_frame", "risk"):
            out[k] = a.get(k)
        if with_frames:
            out["frames"] = [_frame_out(f, request) for f in
                             db().frames.find({"analysis_id": a["_id"]}).sort("order", 1)]
    return out


def _get(aid):
    a = db().analyses.find_one({"_id": aid})
    if not a:
        raise HTTPException(404, f"Analysis {aid} not found")
    return a


def _num(value, name, lo, hi):
    if value is None or str(value).strip() == "":
        return None
    try:
        v = float(value)
    except ValueError:
        raise HTTPException(422, f"{name} must be a number")
    if not lo <= v <= hi:
        raise HTTPException(422, f"{name} must be between {lo} and {hi}")
    return v


# ----------------------------------------------------------------------------- health
@router.get("/health")
def health(request: Request):
    det = request.app.state.detector
    return {"status": "ok", "model": det.info(), "database": settings.MONGO_DB,
            "queue": P._jobs.qsize(), "busy": P._state["busy"]}


# ----------------------------------------------------------------------------- analyses
@router.post("/analyses", status_code=202)
def create_analysis(request: Request,
                    files: List[UploadFile] = File(...),
                    location: str = Form(""),
                    latitude: Optional[str] = Form(None),
                    longitude: Optional[str] = Form(None),
                    water_body: str = Form(""),
                    depth_m: Optional[str] = Form(None),
                    captured_at: str = Form(""),
                    notes: str = Form(""),
                    frame_interval: Optional[str] = Form(None),
                    confidence_threshold: Optional[str] = Form(None)):
    if not files:
        raise HTTPException(422, "Attach at least one photo or video.")
    if len(files) > settings.MAX_FILES:
        raise HTTPException(422, f"At most {settings.MAX_FILES} files per analysis.")
    lat, lon = _num(latitude, "latitude", -90, 90), _num(longitude, "longitude", -180, 180)
    if (lat is None) != (lon is None):
        raise HTTPException(422, "Give both latitude and longitude, or neither.")
    interval = _num(frame_interval, "frame_interval", 0.1, 60) or 1.0
    conf = _num(confidence_threshold, "confidence_threshold", 0.01, 0.99) or settings.DEFAULT_CONFIDENCE

    aid = f"an_{P.now():%Y%m%d_%H%M%S}_{secrets.token_hex(3)}"
    updir = P.analysis_dir(aid) / "uploads"
    updir.mkdir(parents=True, exist_ok=True)
    media, limit = [], settings.MAX_UPLOAD_MB * 1024 * 1024
    try:
        for k, f in enumerate(files):
            name = Path(f.filename or f"file{k}").name
            kind = M.kind_of(name)
            if kind is None:
                raise HTTPException(415, f"{name}: unsupported type. Use JPEG/PNG/WebP/BMP photos or "
                                         f"MP4/WebM/MOV/AVI/MKV video.")
            ext = Path(name).suffix.lower()
            dest, size = updir / f"m{k + 1:02d}{ext}", 0
            with open(dest, "wb") as out:
                while chunk := f.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > limit:
                        raise HTTPException(413, f"{name} is larger than {settings.MAX_UPLOAD_MB} MB.")
                    out.write(chunk)
            item = {"id": f"m{k + 1:02d}", "name": name, "kind": kind, "path": P.rel(dest), "bytes": size,
                    "original_format": f.content_type or M.MIME.get(ext),
                    "model_format": "jpeg" if kind == "image" else "mp4"}
            if kind == "image":
                img = M.read_image(dest)
                item.update(width=img.shape[1], height=img.shape[0], frame_count=1)
            else:
                info = M.video_info(dest)
                if info["duration_seconds"] and info["duration_seconds"] > settings.MAX_VIDEO_SECONDS:
                    raise HTTPException(413, f"{name} is {info['duration_seconds']:.0f} s long; the limit is "
                                             f"{settings.MAX_VIDEO_SECONDS} s.")
                item.update(info, annotated_status="pending", annotated_progress=0)
            media.append(item)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(422, str(e))

    doc = {"_id": aid, "status": "queued", "progress": 0, "message": "Queued", "created_at": P.now(),
           "updated_at": P.now(), "media": media,
           "location_input": {"query": location.strip(), "latitude": lat, "longitude": lon,
                              "water_body": water_body.strip(), "depth_m": _num(depth_m, "depth_m", 0, 11000),
                              "captured_at": captured_at.strip(), "notes": notes.strip()[:1000]},
           "settings": {"frame_interval": interval, "confidence_threshold": conf, "imgsz": settings.IMGSZ},
           "model_info": request.app.state.detector.info()}
    db().analyses.insert_one(doc)
    ahead = P.enqueue_analysis(aid)
    msg = f"Queued - {ahead} job(s) ahead on the GPU" if ahead else "Queued"
    db().analyses.update_one({"_id": aid}, {"$set": {"message": msg}})
    return {"analysis_id": aid, "status": "queued", "progress": 0, "message": msg}


@router.get("/analyses")
def list_analyses(request: Request, limit: int = 20):
    rows = db().analyses.find({}, {"_id": 1}).sort("created_at", -1).limit(max(1, min(limit, 100)))
    return [_analysis_out(_get(r["_id"]), request, with_frames=False) for r in rows]


@router.get("/analyses/{aid}")
def get_analysis(aid: str, request: Request):
    return _analysis_out(_get(aid), request)


@router.get("/analyses/{aid}/frames")
def get_frames(aid: str, request: Request):
    _get(aid)
    return [_frame_out(f, request) for f in db().frames.find({"analysis_id": aid}).sort("order", 1)]


@router.get("/analyses/{aid}/frames/{fid}")
def get_frame(aid: str, fid: str, request: Request):
    f = db().frames.find_one({"_id": f"{aid}/{fid}"})
    if not f:
        raise HTTPException(404, "Frame not found")
    return _frame_out(f, request)


@router.get("/analyses/{aid}/frames/{fid}/annotated")
def frame_annotated(aid: str, fid: str):
    f = db().frames.find_one({"_id": f"{aid}/{fid}"})
    if not f:
        raise HTTPException(404, "Frame not found")
    stem = Path(f["label"].replace(" / ", "_")).stem
    return FileResponse(settings.STORAGE_DIR / f["annotated"], media_type="image/jpeg",
                        filename=f"hydrowatch-{stem}-annotated.jpg")


@router.get("/analyses/{aid}/media/{mid}/annotated")
def media_annotated(aid: str, mid: str):
    a = _get(aid)
    m = next((x for x in a["media"] if x["id"] == mid), None)
    if not m:
        raise HTTPException(404, "Media not found")
    stem = Path(m["name"]).stem
    if m["kind"] == "image":
        f = db().frames.find_one({"analysis_id": aid, "media_id": mid})
        if not f:
            return JSONResponse({"status": a["status"], "message": "Analysis still running"}, status_code=202)
        return FileResponse(settings.STORAGE_DIR / f["annotated"], media_type="image/jpeg",
                            filename=f"hydrowatch-{stem}-annotated.jpg")
    status = m.get("annotated_status", "pending")
    if status == "ready":
        return FileResponse(settings.STORAGE_DIR / m["annotated_path"], media_type="video/mp4",
                            filename=f"hydrowatch-{stem}-annotated.mp4")
    if status in ("pending", "failed") and a["status"] == "completed":
        P.enqueue_render(aid, mid)   # render on demand (first request, or retry after a failure)
        status = "queued"
    return JSONResponse({"status": status, "progress": m.get("annotated_progress", 0),
                         "message": f"Annotated video {status}"}, status_code=202)


@router.get("/analyses/{aid}/report")
def report(aid: str, request: Request, media_id: Optional[str] = None):
    a = _get(aid)
    if a["status"] != "completed":
        raise HTTPException(409, "The analysis has not finished yet.")
    q = {"analysis_id": aid, **({"media_id": media_id} if media_id else {})}
    s = P.summarise(list(db().frames.find(q)))
    debris = [c for c in s["classes"] if c["tier"] == "debris"]
    if debris:
        top = ", ".join(f"{c['label']} ({c['count']})" for c in debris[:3])
        risks = "; ".join(RISK_TEXT[c["class_key"]] for c in debris[:3] if c["class_key"] in RISK_TEXT)
        impact = f"Most frequent debris: {top}. {risks[:1].upper() + risks[1:]}." if risks else f"Most frequent debris: {top}."
    else:
        impact = "No debris was detected in the reviewed media."
    return {"analysis_id": aid, "media_id": media_id, "timestamp": _iso(a.get("completed_at")),
            "location": a.get("location"), "model_info": a.get("model_info"), "settings": a["settings"],
            "media": [_media_out(m) for m in a["media"] if not media_id or m["id"] == media_id], **s,
            "severity": {"category": s["risk"], "methodology": P.RISK_METHOD},
            "impact": {"summary": impact, "methodology": "Rule-based summary of the most frequent debris classes"}}


# ----------------------------------------------------------------------------- hotspots and dashboard
def _hotspot(l, current):
    mean = l["debris_total"] / max(l["frames_count"], 1)
    cats = sorted((l.get("class_counts") or {}).items(), key=lambda kv: -kv[1])
    return {"id": l["_id"], "name": l.get("name") or l.get("label"), "label": l.get("label"),
            "latitude": l["latitude"], "longitude": l["longitude"], "debris_count": l["debris_total"],
            "mean_debris_per_frame": round(mean, 2), "risk": P.risk_of(mean),
            "categories": [CLASS_INFO.get(k, (k,))[0] for k, _ in cats[:4]],
            "analyses_count": l["analyses_count"], "frames_count": l["frames_count"],
            "water_body": l.get("water_body"), "updated_at": _iso(l.get("updated_at")), "is_current": l["_id"] == current}


@router.get("/hotspots")
def hotspots(analysis_id: Optional[str] = None):
    current = P.location_key(_get(analysis_id).get("location")) if analysis_id else None
    rows = [_hotspot(l, current) for l in db().locations.find().sort("debris_total", -1)]
    return sorted(rows, key=lambda h: not h["is_current"])


@router.get("/dashboard")
def dashboard():
    d = db()
    done = list(d.analyses.find({"status": "completed"},
                                {"classes": 1, "debris_detections": 1, "frames_reviewed": 1, "media": 1}))
    debris = Counter()
    for a in done:
        for c in a.get("classes", []):
            if c["tier"] == "debris":
                debris[c["label"]] += c["count"]
    total_debris = sum(debris.values())
    locs = [_hotspot(l, None) for l in d.locations.find()]
    high = sum(1 for h in locs if h["risk"] in ("High", "Critical"))
    frames = sum(a.get("frames_reviewed", 0) for a in done)
    media_n = sum(len(a["media"]) for a in done)
    recent = []
    for a in d.analyses.find().sort("created_at", -1).limit(6):
        loc = a.get("location") or a.get("location_input", {})
        recent.append({"id": a["_id"], "site": loc.get("query") or loc.get("label") or "No location given",
                       "time": _iso(a.get("created_at")), "status": a["status"],
                       "count": f"{a['debris_detections']} debris" if a["status"] == "completed" else a["status"]})
    return {
        "stats": [
            {"label": "Total debris detected", "value": f"{total_debris:,}", "change": "All analyses", "tone": "blue"},
            {"label": "Media analysed", "value": f"{media_n:,}", "change": f"{frames:,} frames reviewed", "tone": "teal"},
            {"label": "Locations monitored", "value": f"{len(locs):,}", "change": "On the map", "tone": "amber"},
            {"label": "High-risk hotspots", "value": f"{high:,}", "change": "High or critical", "tone": "red"},
        ],
        "recent": recent,
        "distribution": [{"name": k, "value": round(100 * v / total_debris, 1)} for k, v in debris.most_common(8)]
        if total_debris else [],
        "totals": {"analyses": len(done), "debris": total_debris, "frames": frames, "locations": len(locs)},
    }
