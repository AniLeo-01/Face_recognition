"""Data models for face detection results."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class DetectedFace:
    """A single detected face with bounding box, confidence, and landmarks."""

    # Bounding box: (x1, y1, x2, y2) in pixel coordinates
    bbox: tuple[float, float, float, float]
    confidence: float
    # Five facial landmarks: left_eye, right_eye, nose, mouth_left, mouth_right
    # Shape: (5, 2) — each row is (x, y)
    landmarks: np.ndarray | None = None

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (
            (self.bbox[0] + self.bbox[2]) / 2,
            (self.bbox[1] + self.bbox[3]) / 2,
        )

    def to_dict(self) -> dict:
        return {
            "bbox": list(self.bbox),
            "confidence": round(float(self.confidence), 4),
            "landmarks": self.landmarks.tolist() if self.landmarks is not None else None,
            "width": round(self.width, 1),
            "height": round(self.height, 1),
        }


@dataclass
class DetectionResult:
    """Result from face detection on a single frame."""

    faces: list[DetectedFace] = field(default_factory=list)
    frame_width: int = 0
    frame_height: int = 0
    inference_time_ms: float = 0.0

    @property
    def num_faces(self) -> int:
        return len(self.faces)

    def to_dict(self) -> dict:
        return {
            "num_faces": self.num_faces,
            "frame_size": [self.frame_width, self.frame_height],
            "inference_time_ms": round(self.inference_time_ms, 2),
            "faces": [f.to_dict() for f in self.faces],
        }
