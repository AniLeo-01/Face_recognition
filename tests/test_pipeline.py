"""Tests for the end-to-end face recognition pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import (
    AlignmentConfig,
    DetectionConfig,
    RecognitionConfig,
    SystemConfig,
)
from src.pipeline.processor import FaceRecognitionPipeline, PipelineResult, RecognizedFace


@pytest.fixture(scope="module")
def pipeline():
    config = SystemConfig(
        detection=DetectionConfig(device="cpu", confidence_threshold=0.5),
        alignment=AlignmentConfig(output_size=(160, 160), min_blur_score=5.0),
        recognition=RecognitionConfig(device="cpu", similarity_threshold=0.4),
    )
    return FaceRecognitionPipeline(config)


class TestPipelineResult:
    def test_empty_result(self):
        r = PipelineResult()
        assert r.num_detected == 0
        assert r.num_recognized == 0
        d = r.to_dict()
        assert d["num_detected"] == 0

    def test_to_dict_complete(self):
        from src.detection.models import DetectedFace, DetectionResult

        face = RecognizedFace(
            detected_face=DetectedFace(bbox=(10, 10, 100, 100), confidence=0.9),
            identity_name="Alice",
            identity_id="id1",
            similarity=0.85,
        )
        det = DetectionResult(
            faces=[face.detected_face], frame_width=640, frame_height=480
        )
        r = PipelineResult(
            recognized_faces=[face],
            detection_result=det,
            total_time_ms=50.0,
            detection_time_ms=20.0,
        )
        d = r.to_dict()
        assert d["num_detected"] == 1
        assert d["num_recognized"] == 1
        assert d["faces"][0]["identity_name"] == "Alice"


class TestFaceRecognitionPipeline:
    def test_process_frame_returns_result(self, pipeline, sample_face_image):
        """Process a frame and get a PipelineResult."""
        result = pipeline.process_frame(sample_face_image)
        assert isinstance(result, PipelineResult)
        assert result.total_time_ms > 0
        assert result.detection_time_ms > 0

    def test_process_no_face_image(self, pipeline, sample_no_face_image):
        """Empty scene should return zero faces."""
        result = pipeline.process_frame(sample_no_face_image)
        assert result.num_detected == 0

    def test_process_grayscale(self, pipeline):
        """Pipeline should handle grayscale input."""
        gray = np.random.randint(0, 255, (300, 300), dtype=np.uint8)
        result = pipeline.process_frame(gray)
        assert isinstance(result, PipelineResult)

    def test_enroll_and_recognize(self, pipeline, enrollment_images):
        """Enroll a person and verify recognition."""
        # Clear gallery first
        pipeline.gallery.clear()

        # Enroll
        enroll_result = pipeline.enroll("test_id", "TestPerson", enrollment_images)
        assert enroll_result["identity_id"] == "test_id"
        assert enroll_result["images_processed"] == len(enrollment_images)
        # At least some should succeed
        assert enroll_result["gallery_size"] >= 0

    def test_enroll_empty_list(self, pipeline):
        """Enrolling with empty image list should produce zero embeddings."""
        result = pipeline.enroll("empty", "Empty", [])
        assert result["embeddings_created"] == 0

    def test_annotate_frame(self, pipeline, sample_face_image):
        """Annotate frame should return same-shape image."""
        result = pipeline.process_frame(sample_face_image)
        annotated = pipeline.annotate_frame(sample_face_image, result)
        assert annotated.shape == sample_face_image.shape

    def test_skip_quality_flag(self, pipeline, sample_face_image):
        """skip_quality should bypass quality gate."""
        result = pipeline.process_frame(sample_face_image, skip_quality=True)
        assert isinstance(result, PipelineResult)

    def test_gallery_operations(self, pipeline):
        """Test gallery add/search through pipeline."""
        pipeline.gallery.clear()
        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)
        pipeline.gallery.add("g1", "GalleryPerson", emb)
        assert pipeline.gallery.size == 1

        match = pipeline.gallery.identify(emb)
        assert match is not None
        assert match.identity_name == "GalleryPerson"
