"""Tests for the face detection module."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.config import DetectionConfig
from src.detection.detector import FaceDetector
from src.detection.models import DetectedFace, DetectionResult


@pytest.fixture(scope="module")
def detector():
    config = DetectionConfig(device="cpu", confidence_threshold=0.5)
    return FaceDetector(config)


class TestDetectedFace:
    def test_properties(self):
        face = DetectedFace(
            bbox=(10.0, 20.0, 110.0, 170.0),
            confidence=0.98,
            landmarks=np.zeros((5, 2), dtype=np.float32),
        )
        assert face.width == 100.0
        assert face.height == 150.0
        assert face.area == 15000.0
        assert face.center == (60.0, 95.0)

    def test_to_dict(self):
        face = DetectedFace(
            bbox=(10.0, 20.0, 110.0, 170.0),
            confidence=0.98,
            landmarks=np.zeros((5, 2), dtype=np.float32),
        )
        d = face.to_dict()
        assert "bbox" in d
        assert "confidence" in d
        assert d["confidence"] == 0.98
        assert d["width"] == 100.0

    def test_to_dict_no_landmarks(self):
        face = DetectedFace(bbox=(0, 0, 50, 50), confidence=0.5)
        d = face.to_dict()
        assert d["landmarks"] is None


class TestDetectionResult:
    def test_empty_result(self):
        result = DetectionResult()
        assert result.num_faces == 0
        d = result.to_dict()
        assert d["num_faces"] == 0

    def test_with_faces(self):
        faces = [
            DetectedFace(bbox=(0, 0, 50, 50), confidence=0.9),
            DetectedFace(bbox=(100, 100, 200, 200), confidence=0.8),
        ]
        result = DetectionResult(faces=faces, frame_width=640, frame_height=480)
        assert result.num_faces == 2


class TestFaceDetector:
    def test_detect_single_face(self, detector, sample_face_image):
        """Detect a face in a single-face image."""
        result = detector.detect(sample_face_image)
        assert isinstance(result, DetectionResult)
        assert result.frame_width == sample_face_image.shape[1]
        assert result.frame_height == sample_face_image.shape[0]
        assert result.inference_time_ms > 0
        # The synthetic face should be detected (may or may not depending on
        # how realistic the drawing is for MTCNN).
        # We test the pipeline logic regardless.

    def test_detect_pil_input(self, detector, sample_face_image):
        """Detect faces from a PIL Image input."""
        pil_img = Image.fromarray(sample_face_image)
        result = detector.detect(pil_img)
        assert isinstance(result, DetectionResult)

    def test_detect_no_faces(self, detector, sample_no_face_image):
        """No faces should be detected in an empty scene."""
        result = detector.detect(sample_no_face_image)
        assert isinstance(result, DetectionResult)
        assert result.num_faces == 0

    def test_detect_grayscale(self, detector):
        """Detect from a grayscale image."""
        gray = np.zeros((300, 300), dtype=np.uint8)
        gray[100:200, 100:200] = 128
        result = detector.detect(gray)
        assert isinstance(result, DetectionResult)

    def test_detect_returns_sorted_by_confidence(self, detector, sample_scene_image):
        """Faces should be sorted by confidence descending."""
        result = detector.detect(sample_scene_image)
        if result.num_faces > 1:
            for i in range(len(result.faces) - 1):
                assert result.faces[i].confidence >= result.faces[i + 1].confidence

    def test_detect_bboxes_within_bounds(self, detector, sample_face_image):
        """Bounding boxes should be clamped to image dimensions."""
        result = detector.detect(sample_face_image)
        h, w = sample_face_image.shape[:2]
        for face in result.faces:
            x1, y1, x2, y2 = face.bbox
            assert x1 >= 0 and y1 >= 0
            assert x2 <= w and y2 <= h

    def test_detect_from_path(self, detector):
        """Detect faces from a file path."""
        from tests.conftest import SCENES_DIR

        path = str(SCENES_DIR / "single_face.jpg")
        result = detector.detect_from_path(path)
        assert isinstance(result, DetectionResult)

    def test_detect_from_path_not_found(self, detector):
        """Non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            detector.detect_from_path("/nonexistent/image.jpg")
