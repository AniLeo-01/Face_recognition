"""Tests for the face alignment and quality assessment modules."""

from __future__ import annotations

import numpy as np
import pytest

from src.alignment.aligner import FaceAligner
from src.alignment.quality import QualityAssessor, QualityReport
from src.config import AlignmentConfig
from src.detection.models import DetectedFace


@pytest.fixture(scope="module")
def aligner():
    return FaceAligner(AlignmentConfig(output_size=(160, 160)))


@pytest.fixture(scope="module")
def quality_assessor():
    return QualityAssessor(AlignmentConfig(min_blur_score=5.0))


def _make_face_with_landmarks(
    cx: int = 150, cy: int = 150, size: int = 100
) -> DetectedFace:
    """Create a DetectedFace with reasonable landmarks."""
    half = size // 2
    return DetectedFace(
        bbox=(cx - half, cy - half, cx + half, cy + half),
        confidence=0.95,
        landmarks=np.array(
            [
                [cx - 25, cy - 15],  # left eye
                [cx + 25, cy - 15],  # right eye
                [cx, cy + 5],  # nose
                [cx - 20, cy + 30],  # mouth left
                [cx + 20, cy + 30],  # mouth right
            ],
            dtype=np.float32,
        ),
    )


class TestFaceAligner:
    def test_align_with_landmarks(self, aligner):
        """Align a face with landmarks produces correct output size."""
        image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        face = _make_face_with_landmarks()
        aligned = aligner.align(image, face)
        assert aligned is not None
        assert aligned.shape == (160, 160, 3)

    def test_align_without_landmarks(self, aligner):
        """Align a face without landmarks falls back to crop+resize."""
        image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        face = DetectedFace(bbox=(50, 50, 250, 250), confidence=0.9)
        aligned = aligner.align(image, face)
        assert aligned is not None
        assert aligned.shape == (160, 160, 3)

    def test_align_multiple(self, aligner):
        """Align multiple faces from one image."""
        image = np.random.randint(0, 255, (400, 600, 3), dtype=np.uint8)
        faces = [
            _make_face_with_landmarks(150, 200, 100),
            _make_face_with_landmarks(400, 200, 90),
        ]
        aligned = aligner.align_multiple(image, faces)
        assert len(aligned) == 2
        for a in aligned:
            assert a.shape == (160, 160, 3)

    def test_align_preserves_dtype(self, aligner):
        """Aligned output should be uint8."""
        image = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
        face = _make_face_with_landmarks()
        aligned = aligner.align(image, face)
        assert aligned.dtype == np.uint8


class TestQualityAssessor:
    def test_good_quality_face(self, quality_assessor):
        """A clear, well-sized face should pass quality."""
        # Create a textured face (high Laplacian variance = not blurry)
        face_crop = np.random.randint(100, 200, (160, 160, 3), dtype=np.uint8)
        face = _make_face_with_landmarks()
        report = quality_assessor.assess(face_crop, face)
        assert isinstance(report, QualityReport)
        assert report.blur_score > 0
        # Random noise has high Laplacian variance, so it should pass blur check
        assert report.passed is True

    def test_blurry_face_rejected(self, quality_assessor):
        """A uniform (blurry) face should fail quality."""
        # Uniform image = zero Laplacian variance
        face_crop = np.full((160, 160, 3), 128, dtype=np.uint8)
        face = _make_face_with_landmarks()
        report = quality_assessor.assess(face_crop, face)
        assert report.blur_score < 1.0
        assert report.passed is False
        assert any("blurry" in r.lower() for r in report.reasons)

    def test_small_face_rejected(self):
        """A face below minimum size should fail."""
        assessor = QualityAssessor(AlignmentConfig(min_face_size=200))
        face_crop = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        report = assessor.assess(face_crop)
        assert report.passed is False
        assert any("small" in r.lower() for r in report.reasons)

    def test_quality_report_to_dict(self, quality_assessor):
        """QualityReport.to_dict() returns all expected keys."""
        face_crop = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        report = quality_assessor.assess(face_crop)
        d = report.to_dict()
        assert "passed" in d
        assert "blur_score" in d
        assert "face_size" in d
        assert "yaw_angle" in d
        assert "pitch_angle" in d

    def test_pose_estimation(self, quality_assessor):
        """Pose estimation from landmarks should return reasonable values."""
        face_crop = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        face = _make_face_with_landmarks()
        report = quality_assessor.assess(face_crop, face)
        # Symmetrical landmarks should give near-zero yaw
        assert abs(report.yaw_angle) < 20.0
