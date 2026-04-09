"""Tests for the FastAPI application and REST endpoints."""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

import src.api.app as app_module
from src.api.app import create_app
from src.config import (
    AlignmentConfig,
    DatabaseConfig,
    DetectionConfig,
    RecognitionConfig,
    SystemConfig,
    set_config,
)
from src.db.repository import IdentityRepository
from src.pipeline.processor import FaceRecognitionPipeline

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "test_samples"


def _make_test_config() -> SystemConfig:
    return SystemConfig(
        detection=DetectionConfig(device="cpu", confidence_threshold=0.5),
        alignment=AlignmentConfig(output_size=(160, 160), min_blur_score=5.0),
        recognition=RecognitionConfig(device="cpu", similarity_threshold=0.4),
        database=DatabaseConfig(url="sqlite+aiosqlite:///./test_api.db"),
    )


@pytest.fixture
async def client():
    config = _make_test_config()
    set_config(config)

    # Manually initialize the pipeline and repository so routes work
    # (ASGITransport doesn't trigger lifespan events)
    app_module._pipeline = FaceRecognitionPipeline(config)
    repo = IdentityRepository(config.database)
    await repo.init_db()
    app_module._repository = repo

    app = create_app(config)
    transport = ASGITransport(app=app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

    # Cleanup
    app_module._pipeline = None
    app_module._repository = None
    await repo.close()
    db_path = Path("./test_api.db")
    if db_path.exists():
        db_path.unlink()


def _image_to_upload_bytes(image: np.ndarray) -> bytes:
    """Convert RGB numpy array to JPEG bytes."""
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", bgr)
    return buf.tobytes()


@pytest.mark.asyncio
async def test_health_endpoint(client):
    """Health endpoint should return system status."""
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "gallery_size" in data


@pytest.mark.asyncio
async def test_config_endpoint(client):
    """Config endpoint should return current settings."""
    resp = await client.get("/config")
    assert resp.status_code == 200
    data = resp.json()
    assert "detection" in data
    assert "alignment" in data
    assert "recognition" in data


@pytest.mark.asyncio
async def test_detect_endpoint(client, sample_face_image):
    """Detect endpoint should accept an image and return results."""
    img_bytes = _image_to_upload_bytes(sample_face_image)
    resp = await client.post(
        "/detect/",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "num_faces" in data
    assert "faces" in data
    assert isinstance(data["faces"], list)


@pytest.mark.asyncio
async def test_recognize_endpoint(client, sample_face_image):
    """Recognize endpoint should run full pipeline."""
    img_bytes = _image_to_upload_bytes(sample_face_image)
    resp = await client.post(
        "/recognize/",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "num_detected" in data
    assert "num_recognized" in data
    assert "timing" in data


@pytest.mark.asyncio
async def test_enroll_endpoint(client, enrollment_images):
    """Enroll endpoint should register a new identity."""
    files = []
    for i, img in enumerate(enrollment_images):
        img_bytes = _image_to_upload_bytes(img)
        files.append(("files", (f"face_{i}.jpg", img_bytes, "image/jpeg")))

    resp = await client.post(
        "/enroll/",
        data={"name": "TestEnroll"},
        files=files,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["identity_name"] == "TestEnroll"
    assert data["images_processed"] == len(enrollment_images)


@pytest.mark.asyncio
async def test_identities_list(client):
    """List identities endpoint should return identity list."""
    resp = await client.get("/identities/")
    assert resp.status_code == 200
    data = resp.json()
    assert "identities" in data
    assert "total" in data


@pytest.mark.asyncio
async def test_events_endpoint(client):
    """Events endpoint should return audit log."""
    resp = await client.get("/events")
    assert resp.status_code == 200
    data = resp.json()
    assert "events" in data


@pytest.mark.asyncio
async def test_detect_empty_file(client):
    """Detect with empty file should return 400."""
    resp = await client.post(
        "/detect/",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_detect_invalid_file(client):
    """Detect with invalid file should return 400."""
    resp = await client.post(
        "/detect/",
        files={"file": ("bad.jpg", b"not-an-image", "image/jpeg")},
    )
    assert resp.status_code == 400
