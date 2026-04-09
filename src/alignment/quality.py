"""Face quality assessment module — blur, size, and pose checks."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from src.config import AlignmentConfig, get_config
from src.detection.models import DetectedFace

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """Quality assessment results for a single face."""

    passed: bool
    blur_score: float
    face_width: float
    face_height: float
    yaw_angle: float  # estimated from landmarks
    pitch_angle: float  # estimated from landmarks
    reasons: list[str]

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "blur_score": round(self.blur_score, 2),
            "face_size": [round(self.face_width, 1), round(self.face_height, 1)],
            "yaw_angle": round(self.yaw_angle, 1),
            "pitch_angle": round(self.pitch_angle, 1),
            "rejection_reasons": self.reasons,
        }


class QualityAssessor:
    """Assess the quality of detected / aligned faces.

    Evaluates blur (Laplacian variance), face size, and estimated
    head pose angles to decide whether a face is usable for recognition.
    """

    def __init__(self, config: AlignmentConfig | None = None) -> None:
        self.config = config or get_config().alignment

    def assess(
        self,
        face_crop: np.ndarray,
        detected_face: DetectedFace | None = None,
    ) -> QualityReport:
        """Assess quality of an aligned face crop.

        Args:
            face_crop: Aligned face image (RGB numpy array).
            detected_face: Original DetectedFace with landmarks (for pose estimation).

        Returns:
            QualityReport with pass/fail and detailed metrics.
        """
        reasons: list[str] = []

        # 1. Blur detection via Laplacian variance
        gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        if blur_score < self.config.min_blur_score:
            reasons.append(
                f"Too blurry (score={blur_score:.1f}, min={self.config.min_blur_score})"
            )

        # 2. Face size check
        face_h, face_w = face_crop.shape[:2]
        if face_w < self.config.min_face_size or face_h < self.config.min_face_size:
            reasons.append(
                f"Face too small ({face_w}x{face_h}, min={self.config.min_face_size})"
            )

        # 3. Pose estimation from landmarks
        yaw, pitch = 0.0, 0.0
        if detected_face is not None and detected_face.landmarks is not None:
            yaw, pitch = self._estimate_pose(detected_face.landmarks)
            if abs(yaw) > self.config.max_yaw_angle:
                reasons.append(
                    f"Yaw too extreme ({yaw:.1f}°, max={self.config.max_yaw_angle}°)"
                )
            if abs(pitch) > self.config.max_pitch_angle:
                reasons.append(
                    f"Pitch too extreme ({pitch:.1f}°, max={self.config.max_pitch_angle}°)"
                )

        return QualityReport(
            passed=len(reasons) == 0,
            blur_score=blur_score,
            face_width=float(face_w),
            face_height=float(face_h),
            yaw_angle=yaw,
            pitch_angle=pitch,
            reasons=reasons,
        )

    @staticmethod
    def _estimate_pose(landmarks: np.ndarray) -> tuple[float, float]:
        """Estimate yaw and pitch from 5-point landmarks.

        Simple geometric estimation:
        - Yaw: asymmetry of eye-to-nose horizontal distances.
        - Pitch: ratio of nose-to-mouth vs eye-to-nose vertical distances.
        """
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        nose = landmarks[2]
        mouth_left = landmarks[3]
        mouth_right = landmarks[4]

        # Eye center
        eye_center = (left_eye + right_eye) / 2.0
        eye_width = np.linalg.norm(right_eye - left_eye)

        if eye_width < 1e-6:
            return 0.0, 0.0

        # Yaw: horizontal asymmetry
        nose_to_left = np.linalg.norm(nose[0] - left_eye[0])
        nose_to_right = np.linalg.norm(nose[0] - right_eye[0])
        yaw_ratio = (nose_to_right - nose_to_left) / (nose_to_right + nose_to_left + 1e-6)
        yaw = float(yaw_ratio * 90.0)  # Rough scaling to degrees

        # Pitch: vertical ratio
        mouth_center = (mouth_left + mouth_right) / 2.0
        eye_nose_dist = nose[1] - eye_center[1]
        nose_mouth_dist = mouth_center[1] - nose[1]

        if abs(eye_nose_dist) > 1e-6:
            pitch_ratio = (nose_mouth_dist / (eye_nose_dist + 1e-6)) - 1.0
            pitch = float(pitch_ratio * 45.0)  # Rough scaling to degrees
        else:
            pitch = 0.0

        return yaw, pitch
