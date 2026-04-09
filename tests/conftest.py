"""Shared test fixtures for the face recognition test suite."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

# Ensure project root is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    AlignmentConfig,
    DatabaseConfig,
    DetectionConfig,
    RecognitionConfig,
    SystemConfig,
)

SAMPLES_DIR = ROOT / "test_samples"
FACES_DIR = SAMPLES_DIR / "faces"
SCENES_DIR = SAMPLES_DIR / "scenes"
ENROLLMENT_DIR = SAMPLES_DIR / "enrollment"


@pytest.fixture(scope="session")
def test_config() -> SystemConfig:
    """Test configuration using CPU and in-memory DB."""
    return SystemConfig(
        detection=DetectionConfig(device="cpu", confidence_threshold=0.5),
        alignment=AlignmentConfig(output_size=(160, 160), min_blur_score=5.0),
        recognition=RecognitionConfig(device="cpu", similarity_threshold=0.4),
        database=DatabaseConfig(url="sqlite+aiosqlite:///./test_face_recognition.db"),
    )


@pytest.fixture(scope="session")
def sample_face_image() -> np.ndarray:
    """Load a single face sample image (RGB)."""
    path = FACES_DIR / "person_A" / "face_00.jpg"
    img = cv2.imread(str(path))
    assert img is not None, f"Sample face not found at {path}"
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@pytest.fixture(scope="session")
def sample_scene_image() -> np.ndarray:
    """Load a multi-face scene image (RGB)."""
    path = SCENES_DIR / "two_faces.jpg"
    img = cv2.imread(str(path))
    assert img is not None, f"Sample scene not found at {path}"
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@pytest.fixture(scope="session")
def sample_no_face_image() -> np.ndarray:
    """Load an image with no faces (RGB)."""
    path = SCENES_DIR / "no_faces.jpg"
    img = cv2.imread(str(path))
    assert img is not None, f"No-face sample not found at {path}"
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@pytest.fixture(scope="session")
def enrollment_images() -> list[np.ndarray]:
    """Load enrollment images for a single identity."""
    paths = sorted(ENROLLMENT_DIR.glob("*.jpg"))
    images = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is not None:
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    assert len(images) > 0, "No enrollment images found"
    return images


@pytest.fixture(scope="session")
def person_images() -> dict[str, list[np.ndarray]]:
    """Load images for all test persons."""
    persons: dict[str, list[np.ndarray]] = {}
    for person_dir in sorted(FACES_DIR.iterdir()):
        if person_dir.is_dir():
            imgs = []
            for p in sorted(person_dir.glob("*.jpg")):
                img = cv2.imread(str(p))
                if img is not None:
                    imgs.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if imgs:
                persons[person_dir.name] = imgs
    return persons


@pytest.fixture
def cleanup_db():
    """Clean up test database after test."""
    yield
    db_path = Path("./test_face_recognition.db")
    if db_path.exists():
        db_path.unlink()
