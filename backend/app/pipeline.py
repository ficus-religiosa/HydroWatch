"""One background worker owns the GPU and processes jobs in order:
  analysis  - run the detector on every photo and on sampled video frames (fast; drives the progress bar)
  render    - draw boxes on EVERY frame of a video and write a browser-playable MP4 (runs after its analysis)
"""
import math
import queue
import threading
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from . import media as M
from .config import settings
from .db import db
from .detector import CLASS_INFO, draw
from .geocode import geocode

_jobs = queue.Queue()
_state = {"detector": None, "busy": False}

RISK_METHOD = ("Prototype scale from mean debris detections per reviewed photo/frame: "
               "Low < 1, Moderate < 3, High < 6, Critical >= 6. A relative indicator, not a calibrated "
               "physical density (items per square metre would need the camera's distance to the seabed).")


def now():
    return datetime.now(timezone.utc)


def risk_of(mean_debris):
    return "Low" if mean_debris < 1 else "Moderate" if mean_debris < 3 else "High" if mean_debris < 6 else "Critical"


def analysis_dir(aid):
    return settings.STORAGE_DIR / "analyses" / aid


def rel(path):
    return Path(path).resolve().relative_to(settings.STORAGE_DIR.resolve()).as_posix()


# ----------------------------------------------------------------------------- worker
def start_worker(detector):
    _state["detector"] = detector
    threading.Thread(target=_loop, daemon=True, name="hydrowatch-gpu").start()


def _loop():
    while True:
        kind, args = _jobs.get()
        _state["busy"] = True
        try:
            (run_analysis if kind == "analysis" else render_video)(*args)
        except Exception:
            traceback.print_exc()
        finally:
            _state["busy"] = False
            _jobs.task_done()


def enqueue_analysis(aid):
    ahead = _jobs.qsize() + (1 if _state["busy"] else 0)
    _jobs.put(("analysis", (aid,)))
    return ahead


def enqueue_render(aid, mid):
    _set_media(aid, mid, annotated_status="queued", annotated_progress=0)
    _jobs.put(("render", (aid, mid)))


def _set(aid, **fields):
    db().analyses.update_one({"_id": aid}, {"$set": {**fields, "updated_at": now()}})


def _set_media(aid, mid, **fields):
    db().analyses.update_one({"_id": aid, "media.id": mid}, {"$set": {f"media.$.{k}": v for k, v in fields.items()}})


# ----------------------------------------------------------------------------- location
def resolve_location(a):
    li = a.get("location_input", {})
    q = (li.get("query") or "").strip()
    loc = {"query": q, "label": q, "latitude": None, "longitude": None, "source": None,
           "water_body": li.get("water_body") or None, "depth_m": li.get("depth_m"),
           "captured_at": li.get("captured_at") or None, "notes": li.get("notes") or None}
    if li.get("latitude") is not None:
        loc.update(latitude=li["latitude"], longitude=li["longitude"], source="entered coordinates")
    elif q:
        g = geocode(q)
        if g:
            loc.update(label=g["label"], latitude=g["latitude"], longitude=g["longitude"], source="place name lookup")
    if loc["latitude"] is None:
        for m in a["media"]:
            if m["kind"] == "image":
                gps = M.exif_gps(settings.STORAGE_DIR / m["path"])
                if gps:
                    loc.update(latitude=gps[0], longitude=gps[1], source="photo GPS")
                    break
    if not loc["label"] and loc["latitude"] is not None:
        loc["label"] = f"{loc['latitude']:.4f}, {loc['longitude']:.4f}"
    return loc


# ----------------------------------------------------------------------------- analysis
def _save_frame(aid, m, order, frame_number, fps, img, dets):
    fdir = analysis_dir(aid) / "frames"
    if m["kind"] == "image":
        fid, label, ts = f"{m['id']}-photo", m["name"], None
    else:
        fid = f"{m['id']}-frame-{frame_number:06d}"
        ts = round(frame_number / fps, 2)
        label = f"{m['name']} / frame {frame_number}"
    preview, annotated = fdir / f"{fid}.jpg", fdir / f"{fid}-annotated.jpg"
    M.write_jpeg(preview, img, quality=88, max_side=settings.PREVIEW_MAX_SIDE)
    M.write_jpeg(annotated, draw(img, dets), quality=90)
    dets = [{**d, "id": f"{fid}-d{k + 1:03d}"} for k, d in enumerate(dets)]
    n_debris = sum(1 for d in dets if d["tier"] == "debris")
    h, w = img.shape[:2]
    db().frames.insert_one({
        "_id": f"{aid}/{fid}", "id": fid, "analysis_id": aid, "media_id": m["id"], "order": order,
        "kind": m["kind"], "frame_number": frame_number, "timestamp_seconds": ts, "label": label,
        "width": w, "height": h, "preview": rel(preview), "annotated": rel(annotated),
        "detections": dets, "debris_count": n_debris,
        "density": {"value": n_debris, "unit": "debris in frame"},
    })


def run_analysis(aid):
    det = _state["detector"]
    try:
        a = db().analyses.find_one({"_id": aid})
        conf, interval = a["settings"]["confidence_threshold"], a["settings"]["frame_interval"]
        _set(aid, status="processing", progress=1, message="Resolving location")
        _set(aid, location=resolve_location(a))

        steps, units = {}, 0
        for m in a["media"]:
            if m["kind"] == "image":
                units += 1
            else:
                steps[m["id"]] = max(1, round(interval * m["fps"]))
                units += max(1, math.ceil((m["frame_count"] or 1) / steps[m["id"]]))
        done, order = 0, 0

        def progress(msg):
            _set(aid, progress=min(99, 2 + round(97 * done / max(units, 1))), message=msg)

        for m in a["media"]:
            src = settings.STORAGE_DIR / m["path"]
            if m["kind"] == "image":
                progress(f"Analysing {m['name']}")
                img = M.read_image(src)
                _save_frame(aid, m, order, None, None, img, det.predict([img], conf)[0])
                order, done = order + 1, done + 1
                continue
            step, batch = steps[m["id"]], []
            total = max(1, math.ceil((m["frame_count"] or 1) / step))

            def flush():
                nonlocal order, done
                for (i, fr), dets in zip(batch, det.predict([fr for _, fr in batch], conf)):
                    _save_frame(aid, m, order, i, m["fps"], fr, dets)
                    order, done = order + 1, done + 1
                progress(f"{m['name']}: frame {min(done, units)} of {units} reviewed")
                batch.clear()

            progress(f"{m['name']}: sampling one frame every {interval:g} s ({total} frames)")
            for i, fr in M.iter_frames(src):
                if i % step == 0:
                    batch.append((i, fr))
                    if len(batch) >= settings.BATCH:
                        flush()
            if batch:
                flush()

        summary = summarise(list(db().frames.find({"analysis_id": aid})))
        _set(aid, status="completed", progress=100, completed_at=now(),
             message=f"Reviewed {summary['frames_reviewed']} photos/frames", **summary)
        _update_location_totals(aid)
        if settings.RENDER_VIDEOS:
            for m in a["media"]:
                if m["kind"] == "video":
                    enqueue_render(aid, m["id"])
    except Exception as e:
        traceback.print_exc()
        _set(aid, status="failed", message=f"Analysis failed: {e}")


def summarise(frames):
    dets = [d for f in frames for d in f["detections"]]
    by_key = Counter(d["class_key"] for d in dets)
    classes = [{"label": CLASS_INFO.get(k, (k, "debris"))[0], "class_key": k,
                "tier": CLASS_INFO.get(k, (k, "debris"))[1], "count": n} for k, n in by_key.most_common()]
    sizes = Counter(d["size"].lower() for d in dets)
    debris = sum(f["debris_count"] for f in frames)
    mean = debris / len(frames) if frames else 0.0
    return {
        "total_detections": len(dets),
        "debris_detections": debris,
        "average_confidence": round(100 * sum(d["confidence"] for d in dets) / len(dets), 1) if dets else 0,
        "classes": classes,
        "size_stats": {k: sizes.get(k, 0) for k in ("small", "medium", "large")},
        "frames_reviewed": len(frames),
        "density": {"value": round(mean, 2), "unit": "debris per frame"},
        "peak_debris_per_frame": max((f["debris_count"] for f in frames), default=0),
        "risk": risk_of(mean),
    }


def location_key(loc):
    if not loc or loc.get("latitude") is None:
        return None
    return f"{loc['latitude']:.3f},{loc['longitude']:.3f}"   # ~100 m grid: repeat surveys of a spot add up


def _update_location_totals(aid):
    a = db().analyses.find_one({"_id": aid})
    loc, key = a.get("location"), location_key(a.get("location"))
    if key is None:
        return
    inc = {"analyses_count": 1, "frames_count": a["frames_reviewed"], "debris_total": a["debris_detections"],
           "detections_total": a["total_detections"]}
    for c in a["classes"]:
        if c["tier"] == "debris":
            inc[f"class_counts.{c['class_key']}"] = c["count"]
    db().locations.update_one(
        {"_id": key},
        {"$inc": inc,
         "$set": {"name": loc["query"] or loc["label"], "label": loc["label"], "latitude": loc["latitude"],
                  "longitude": loc["longitude"], "water_body": loc.get("water_body"),
                  "last_analysis_id": aid, "updated_at": now()},
         "$setOnInsert": {"created_at": now()}},
        upsert=True)


# ----------------------------------------------------------------------------- annotated video
def render_video(aid, mid):
    det = _state["detector"]
    try:
        a = db().analyses.find_one({"_id": aid})
        m = next(x for x in a["media"] if x["id"] == mid)
        conf = a["settings"]["confidence_threshold"]
        _set_media(aid, mid, annotated_status="rendering", annotated_progress=0)
        out = analysis_dir(aid) / "annotated" / f"{mid}-annotated.mp4"
        writer = M.Mp4Writer(out, m["width"], m["height"], m["fps"])
        n, batch, last = max(1, m["frame_count"] or 1), [], -1

        def flush(done):
            nonlocal last
            for fr, dets in zip(batch, det.predict(batch, conf)):
                writer.write(draw(fr, dets))
            batch.clear()
            pct = min(99, round(100 * done / n))
            if pct >= last + 5:
                _set_media(aid, mid, annotated_progress=pct)
                last = pct

        try:
            for i, fr in M.iter_frames(settings.STORAGE_DIR / m["path"]):
                batch.append(fr)
                if len(batch) >= settings.BATCH:
                    flush(i + 1)
            if batch:
                flush(n)
        finally:
            writer.close()
        _set_media(aid, mid, annotated_status="ready", annotated_progress=100, annotated_path=rel(out))
    except Exception as e:
        traceback.print_exc()
        _set_media(aid, mid, annotated_status="failed", annotated_error=str(e))
