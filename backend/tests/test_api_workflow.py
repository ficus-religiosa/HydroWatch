import io
import time
import tempfile
from pathlib import Path
import cv2
import numpy as np

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.services.geocoding_service import geocoding_service
from app.repository.analysis_repository import FileAnalysisRepository
from app.storage.cleanup import recover_interrupted_jobs
from app.schemas.analysis import AnalysisResponse
from app.core import auth
from app.core.config import settings


def make_image() -> io.BytesIO:
    stream = io.BytesIO()
    Image.new("RGB", (40, 30), "green").save(stream, format="PNG")
    stream.seek(0)
    return stream


def wait_for_completion(client: TestClient, analysis_id: str) -> dict:
    for _ in range(60):
        result = client.get(f"/api/v1/analyses/{analysis_id}").json()
        if result["status"] in {"completed", "failed"}:
            return result
        time.sleep(0.02)
    raise AssertionError("analysis did not finish")


def wait_for_annotation(client: TestClient, analysis_id: str, media_id: str):
    responses = []
    for _ in range(100):
        response = client.get(f"/api/v1/analyses/{analysis_id}/media/{media_id}/annotated")
        responses.append(response.status_code)
        if response.status_code == 200:
            return responses, response
        assert response.status_code == 202
        time.sleep(0.02)
    raise AssertionError(f"annotation did not finish: {responses}")


def test_health_and_invalid_analysis_errors():
    with TestClient(app) as client:
        assert client.get("/api/v1/health").status_code == 200
        response = client.get("/api/v1/analyses/analysis_missing")
        assert response.status_code == 404
        assert set(response.json()) == {"error"}
        assert "message" in response.json()["error"]


def test_image_upload_frames_filters_annotation_hotspots_and_report(monkeypatch):
    async def fake_geocode(query):
        return {"location": {"query": query, "label": "Test Harbor", "latitude": 12.3, "longitude": 45.6}, "warning": None}

    monkeypatch.setattr(geocoding_service, "geocode", fake_geocode)
    with TestClient(app) as client:
        accepted = client.post(
            "/api/v1/analyses",
            data={"location": "Test Harbor"},
            files={"files": ("capture.png", make_image(), "image/png")},
        )
        assert accepted.status_code == 202
        analysis = wait_for_completion(client, accepted.json()["analysis_id"])
        assert analysis["status"] == "completed"
        assert analysis["location"]["latitude"] == 12.3
        assert analysis["model_info"]["is_mock"] is True
        assert all(0 <= item["confidence"] <= 1 for item in analysis["frames"][0]["detections"])
        assert all(0 <= item["bbox"][key] <= 1 for item in analysis["frames"][0]["detections"] for key in ("x", "y", "width", "height"))

        analysis_id = analysis["analysis_id"]
        frame = analysis["frames"][0]
        assert client.get(f"/api/v1/analyses/{analysis_id}/frames?media_id=media_001").status_code == 200
        assert client.get(f"/api/v1/analyses/{analysis_id}/frames?media_id=media_missing").status_code == 404
        annotated = client.get(f"/api/v1/analyses/{analysis_id}/frames/{frame['id']}/annotated")
        assert annotated.status_code == 200
        assert annotated.headers["content-type"].startswith("image/jpeg")
        assert "attachment" in annotated.headers["content-disposition"]
        hotspots = client.get(f"/api/v1/hotspots?analysis_id={analysis_id}")
        assert hotspots.status_code == 200
        assert hotspots.json()[0]["count"] == analysis["total_detections"]
        report = client.get(f"/api/v1/analyses/{analysis_id}/report?media_id=media_001")
        assert report.status_code == 200
        assert report.json()["severity"]["methodology"] == "prototype"
        assert report.json()["impact"]["methodology"] == "prototype"


def test_invalid_image_is_rejected_after_decode():
    with TestClient(app) as client:
        accepted = client.post(
            "/api/v1/analyses",
            files={"files": ("corrupt.png", io.BytesIO(b"not-an-image"), "image/png")},
        )
        analysis = wait_for_completion(client, accepted.json()["analysis_id"])
        assert analysis["status"] == "failed"
        assert "corrupt" in analysis["message"].lower() or "unreadable" in analysis["message"].lower()


def test_unresolved_location_has_warning_and_no_coordinates(monkeypatch):
    async def no_match(query):
        return {"location": None, "warning": {"code": "location_not_found", "message": "No match."}}

    monkeypatch.setattr(geocoding_service, "geocode", no_match)
    with TestClient(app) as client:
        accepted = client.post("/api/v1/analyses", data={"location": "Unknown Place"}, files={"files": ("capture.png", make_image(), "image/png")})
        result = wait_for_completion(client, accepted.json()["analysis_id"])
        assert result["location"] is None
        assert result["warning"]["code"] == "location_not_found"
        assert client.get(f"/api/v1/hotspots?analysis_id={result['analysis_id']}").json() == []


def test_processing_records_are_recovered_after_restart(tmp_path):
    repository = FileAnalysisRepository(tmp_path)
    repository.save(AnalysisResponse(analysis_id="analysis_01JRECOVER", status="processing"))
    recover_interrupted_jobs(repository)
    result = repository.get("analysis_01JRECOVER")
    assert result.status == "failed"
    assert result.message == "Interrupted by server restart"


def test_annotated_video_202_to_200_and_concurrent_cache():
    with tempfile.TemporaryDirectory() as directory:
        source = Path(directory) / "sample.mp4"
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (32, 24))
        for index in range(10):
            writer.write(np.full((24, 32, 3), index * 10, dtype=np.uint8))
        writer.release()
        with TestClient(app) as client:
            accepted = client.post("/api/v1/analyses", files={"files": ("sample.mp4", source.open("rb"), "video/mp4")})
            analysis = wait_for_completion(client, accepted.json()["analysis_id"])
            media_id = analysis["media"][0]["id"]
            first = client.get(f"/api/v1/analyses/{analysis['analysis_id']}/media/{media_id}/annotated")
            assert first.status_code in {202, 200}
            responses = [client.get(f"/api/v1/analyses/{analysis['analysis_id']}/media/{media_id}/annotated") for _ in range(3)]
            assert all(response.status_code in {202, 200} for response in responses)
            statuses, final = wait_for_annotation(client, analysis["analysis_id"], media_id)
            assert 202 in statuses or first.status_code == 200
            assert final.headers["content-type"].startswith("video/mp4")


def test_upload_validation_and_media_path_security():
    with TestClient(app) as client:
        unsupported = client.post("/api/v1/analyses", files={"files": ("file.txt", io.BytesIO(b"text"), "text/plain")})
        assert unsupported.status_code == 202
        failed = wait_for_completion(client, unsupported.json()["analysis_id"])
        assert failed["status"] == "failed"
        assert "stack" not in failed["message"].lower()
        assert client.get("/static/media/../app/main.py").status_code == 404
        assert client.get("/static/media/%2e%2e/app/main.py").status_code == 404
        assert client.get("/static/media/C:\\Windows\\win.ini").status_code == 404


def test_health_is_public_and_route_auth_dependency_is_present():
    with TestClient(app) as client:
        assert client.get("/api/v1/health").status_code == 200
    analysis_route = next(route for route in app.routes if route.path == "/api/v1/analyses")
    hotspot_route = next(route for route in app.routes if route.path == "/api/v1/hotspots")
    health_route = next(route for route in app.routes if route.path == "/api/v1/health")
    assert analysis_route.dependencies
    assert hotspot_route.dependencies
    assert not health_route.dependencies


def test_preview_url_uses_public_base_or_request_base(monkeypatch):
    with TestClient(app, base_url="http://testserver:8765") as client:
        original = settings.PUBLIC_BASE_URL
        monkeypatch.setattr(settings, "PUBLIC_BASE_URL", "https://public.example")
        accepted = client.post("/api/v1/analyses", files={"files": ("capture.png", make_image(), "image/png")})
        result = wait_for_completion(client, accepted.json()["analysis_id"])
        assert result["frames"][0]["preview_url"].startswith("https://public.example/")
        monkeypatch.setattr(settings, "PUBLIC_BASE_URL", "")
        accepted = client.post("/api/v1/analyses", files={"files": ("capture.png", make_image(), "image/png")})
        result = wait_for_completion(client, accepted.json()["analysis_id"])
        assert result["frames"][0]["preview_url"].startswith("http://testserver:8765/")
        monkeypatch.setattr(settings, "PUBLIC_BASE_URL", original)