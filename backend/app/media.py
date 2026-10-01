"""Reading and writing photos and videos. Paths may contain non-ASCII characters on Windows,
so images go through numpy buffers instead of cv2.imread / cv2.imwrite."""
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXT = {".mp4", ".webm", ".mov", ".avi", ".mkv"}
MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp",
        ".bmp": "image/bmp", ".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime",
        ".avi": "video/x-msvideo", ".mkv": "video/x-matroska"}


def kind_of(filename):
    ext = Path(filename).suffix.lower()
    return "image" if ext in IMAGE_EXT else "video" if ext in VIDEO_EXT else None


def read_image(path):
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"{Path(path).name} could not be read as an image")
    return img


def write_jpeg(path, img, quality=90, max_side=None):
    if max_side and max(img.shape[:2]) > max_side:
        s = max_side / max(img.shape[:2])
        img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    buf.tofile(str(path))


def exif_gps(path):
    """(lat, lon) from a photo's EXIF GPS tags, or None."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            gps = im.getexif().get_ifd(0x8825)
        if not gps or 2 not in gps or 4 not in gps:
            return None

        def deg(v):
            d, m, s = (float(x) for x in v)
            return d + m / 60 + s / 3600

        lat, lon = deg(gps[2]), deg(gps[4])
        if gps.get(1) in ("S", b"S"):
            lat = -lat
        if gps.get(3) in ("W", b"W"):
            lon = -lon
        return (round(lat, 6), round(lon, 6)) if -90 <= lat <= 90 and -180 <= lon <= 180 else None
    except Exception:
        return None


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"{Path(path).name} could not be opened as a video")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    fps = fps if 1 <= fps <= 240 else 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"fps": fps, "frame_count": n, "width": w, "height": h, "duration_seconds": round(n / fps, 2) if n else None}


def iter_frames(path):
    """Yield (frame_index, BGR frame) for every frame."""
    cap = cv2.VideoCapture(str(path))
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield i, frame
            i += 1
    finally:
        cap.release()


class Mp4Writer:
    """Browser-playable H.264 MP4 via the ffmpeg bundled with imageio-ffmpeg.
    Falls back to OpenCV's mp4v codec (downloads fine, but browsers may not play it inline)."""

    def __init__(self, path, width, height, fps):
        self.path, self.fps = str(path), fps
        self.w, self.h = width - width % 2, height - height % 2   # H.264 needs even dimensions
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.gen, self.cv = None, None
        try:
            import imageio_ffmpeg
            self.gen = imageio_ffmpeg.write_frames(
                self.path, (self.w, self.h), pix_fmt_in="bgr24", pix_fmt_out="yuv420p", fps=fps,
                codec="libx264", macro_block_size=2, quality=None, bitrate=None,
                output_params=["-crf", "23", "-preset", "veryfast", "-movflags", "+faststart"])
            self.gen.send(None)
        except Exception:
            self.gen = None
            self.cv = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (self.w, self.h))

    def write(self, frame):
        frame = np.ascontiguousarray(frame[: self.h, : self.w])
        if self.gen is not None:
            self.gen.send(frame.tobytes())
        else:
            self.cv.write(frame)

    def close(self):
        if self.gen is not None:
            self.gen.close()
        if self.cv is not None:
            self.cv.release()
