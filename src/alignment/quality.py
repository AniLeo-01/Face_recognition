"""Face quality assessment module — multi-metric, region-weighted, soft-gate approach.

Design goals
------------
* **Never hard-reject a face that carries useful identity signal.**
  Instead of a binary pass/fail, every face gets a continuous quality_score (0–100)
  that flows downstream to adjust the matching threshold.

* **Region-weighted blur** — the eye strip is the most discriminative zone for
  recognition; we weight its sharpness 3× relative to the rest of the crop.

* **Scale-normalised pose estimation** — all landmark distances are normalised
  by the inter-eye distance, so the estimator is invariant to face size and
  image resolution.  We also estimate in-plane roll so extreme head-tilts are
  caught correctly.

* **Three quality tiers**:
    PASS     – quality_score ≥ hard_quality_threshold  → match with normal threshold
    MARGINAL – quality_score ≥ soft_quality_threshold  → match with tightened threshold
    FAIL     – quality_score <  soft_quality_threshold → skip (too degraded)

Quality score formula
---------------------
  quality_score = 0.40 * blur_component
                + 0.35 * pose_component
                + 0.15 * size_component
                + 0.10 * symmetry_component
  (all components are in [0, 100])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from src.config import AlignmentConfig, get_config
from src.detection.models import DetectedFace

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tier enum
# ---------------------------------------------------------------------------

class QualityTier(str, Enum):
    PASS     = "pass"      # quality_score >= hard threshold
    MARGINAL = "marginal"  # quality_score >= soft threshold  (embedded, stricter match)
    FAIL     = "fail"      # quality_score <  soft threshold  (skip embedding)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class QualityReport:
    """Quality assessment results for a single face."""

    # Overall
    quality_score: float      # 0–100 composite score
    tier: QualityTier         # PASS / MARGINAL / FAIL

    # Individual components (0–100 each)
    blur_score: float         # combined region-weighted sharpness
    pose_score: float         # 100 = perfectly frontal
    size_score: float         # 100 = large face
    symmetry_score: float     # 100 = fully symmetric

    # Estimated pose angles (degrees)
    yaw_angle: float
    pitch_angle: float
    roll_angle: float

    # Face size in the crop (pixels)
    face_width: float
    face_height: float

    # Diagnostics
    reasons: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True for PASS and MARGINAL tiers."""
        return self.tier in (QualityTier.PASS, QualityTier.MARGINAL)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "tier": self.tier.value,
            "quality_score": round(self.quality_score, 1),
            "blur_score": round(self.blur_score, 2),
            "pose_score": round(self.pose_score, 1),
            "size_score": round(self.size_score, 1),
            "symmetry_score": round(self.symmetry_score, 1),
            "face_size": [round(self.face_width, 1), round(self.face_height, 1)],
            "yaw_angle": round(self.yaw_angle, 1),
            "pitch_angle": round(self.pitch_angle, 1),
            "roll_angle": round(self.roll_angle, 1),
            "rejection_reasons": self.reasons,
        }


# ---------------------------------------------------------------------------
# Assessor
# ---------------------------------------------------------------------------

class QualityAssessor:
    """Assess face quality using a multi-metric, region-weighted approach.

    Each metric returns a 0–100 score; they are combined into a composite
    quality_score that determines the processing tier (PASS / MARGINAL / FAIL).
    """

    def __init__(self, config: AlignmentConfig | None = None) -> None:
        self.config = config or get_config().alignment

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def assess(
        self,
        face_crop: np.ndarray,
        detected_face: DetectedFace | None = None,
    ) -> QualityReport:
        """Compute quality metrics and assign a processing tier.

        Args:
            face_crop:     Aligned face image (RGB numpy array, any size).
            detected_face: Original DetectedFace with 5-point landmarks.

        Returns:
            QualityReport with composite score, tier, and per-metric breakdown.
        """
        reasons: list[str] = []

        # ── 1. Blur ──────────────────────────────────────────────────────────
        blur_score = self._region_weighted_blur(face_crop)

        # ── 2. Face size ─────────────────────────────────────────────────────
        face_h, face_w = face_crop.shape[:2]
        min_dim = self.config.min_face_size
        if face_w < min_dim or face_h < min_dim:
            size_norm = min(face_w, face_h) / min_dim
        else:
            # Saturate at 4× min_dim → 100
            size_norm = min(1.0, min(face_w, face_h) / (min_dim * 4))
        size_score = float(size_norm * 100)

        # ── 3. Pose ───────────────────────────────────────────────────────────
        yaw, pitch, roll, symmetry = 0.0, 0.0, 0.0, 100.0
        if detected_face is not None and detected_face.landmarks is not None:
            yaw, pitch, roll, symmetry = self._estimate_pose(detected_face.landmarks)

        pose_score = self._pose_score(yaw, pitch, roll)

        # ── 4. Composite score ────────────────────────────────────────────────
        # Weights: blur 40 %, pose 35 %, size 15 %, symmetry 10 %
        quality_score = (
            0.40 * blur_score
            + 0.35 * pose_score
            + 0.15 * size_score
            + 0.10 * symmetry
        )
        quality_score = float(np.clip(quality_score, 0.0, 100.0))

        # ── 5. Tier assignment ────────────────────────────────────────────────
        hard_thr = self.config.hard_quality_threshold
        soft_thr = self.config.soft_quality_threshold

        if quality_score >= hard_thr:
            tier = QualityTier.PASS
        elif quality_score >= soft_thr:
            tier = QualityTier.MARGINAL
            reasons.append(
                f"Marginal quality ({quality_score:.1f}), matched with tighter threshold"
            )
        else:
            tier = QualityTier.FAIL
            reasons.append(
                f"Quality too low ({quality_score:.1f} < {soft_thr})"
            )

        # Append per-metric diagnostics for transparency
        if blur_score < 40:
            reasons.append(f"Low sharpness (blur_score={blur_score:.1f})")
        if abs(yaw) > self.config.max_yaw_angle:
            reasons.append(f"Yaw too extreme ({yaw:.1f}°, max={self.config.max_yaw_angle}°)")
        if abs(pitch) > self.config.max_pitch_angle:
            reasons.append(f"Pitch too extreme ({pitch:.1f}°, max={self.config.max_pitch_angle}°)")
        if abs(roll) > 35:
            reasons.append(f"Roll too extreme ({roll:.1f}°, threshold=35°)")

        return QualityReport(
            quality_score=quality_score,
            tier=tier,
            blur_score=blur_score,
            pose_score=pose_score,
            size_score=size_score,
            symmetry_score=symmetry,
            yaw_angle=yaw,
            pitch_angle=pitch,
            roll_angle=roll,
            face_width=float(face_w),
            face_height=float(face_h),
            reasons=reasons,
        )

    # ------------------------------------------------------------------
    # Blur assessment — region-weighted
    # ------------------------------------------------------------------

    def _region_weighted_blur(self, face_crop: np.ndarray) -> float:
        """Compute a region-weighted sharpness score (0–100).

        The eye strip (top 55 % of the crop) carries 3× the weight of the
        lower face, because eyes contribute most to embedding quality.
        Two complementary metrics are combined:
          * Laplacian variance    — sensitive to defocus
          * Tenengrad (Sobel mag) — sensitive to fine edge detail
        """
        # Keep as uint8 — OpenCV's Laplacian/Sobel require uint8 (or int16/int32)
        # source when the destination depth is CV_64F.  Converting to float32
        # first would trigger a "Unsupported combination of source/dest format"
        # error in OpenCV ≥ 4.x on some platforms (macOS ARM in particular).
        gray = cv2.cvtColor(face_crop, cv2.COLOR_RGB2GRAY)   # uint8, shape (H, W)
        h = gray.shape[0]
        split = int(h * 0.55)

        eye_region   = gray[:split, :]
        lower_region = gray[split:, :]

        def _metrics(patch: np.ndarray) -> float:
            # patch is uint8; CV_64F output is fine from uint8 source on all platforms
            lap = float(cv2.Laplacian(patch, cv2.CV_64F).var())
            # Tenengrad (Sobel gradient energy)
            gx  = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
            gy  = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
            ten = float(np.mean(gx ** 2 + gy ** 2))
            # Geometric mean keeps the two metrics on the same scale
            return float(np.sqrt(max(lap, 0) * max(ten, 0) + 1e-6))

        score_eye   = _metrics(eye_region)
        score_lower = _metrics(lower_region)

        # Weighted combination (eye region 3×)
        raw = (3 * score_eye + score_lower) / 4.0

        # Normalise to 0–100:
        # empirically ~ 50 raw ≈ "acceptable", 2000 raw ≈ "excellent"
        # use log scale to avoid extreme values dominating
        normalised = float(np.clip(np.log1p(raw) / np.log1p(2000) * 100, 0, 100))
        return normalised

    # ------------------------------------------------------------------
    # Pose estimation — scale-normalised, multi-ratio
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_pose(
        landmarks: np.ndarray,
    ) -> tuple[float, float, float, float]:
        """Estimate (yaw, pitch, roll, symmetry_score) from 5-point MTCNN landmarks.

        All distances are normalised by the inter-eye distance (IED) so the
        estimator is invariant to face size and image resolution.

        Landmark order (MTCNN): left_eye, right_eye, nose, mouth_left, mouth_right.

        Returns
        -------
        yaw       – signed degrees; positive = face turned right
        pitch     – signed degrees; positive = face tilted up
        roll      – signed degrees; in-plane rotation
        symmetry  – score 0–100 (100 = perfectly symmetric)
        """
        le, re, nose, ml, mr = (landmarks[i].astype(float) for i in range(5))

        # ── Inter-eye distance ────────────────────────────────────────────────
        ied = float(np.linalg.norm(re - le))
        if ied < 1e-4:
            return 0.0, 0.0, 0.0, 100.0

        eye_center  = (le + re) / 2.0
        mouth_center = (ml + mr) / 2.0

        # ── Roll (in-plane rotation) ──────────────────────────────────────────
        # Angle of the eye-line relative to horizontal
        dx = re[0] - le[0]
        dy = re[1] - le[1]
        roll = float(np.degrees(np.arctan2(dy, dx)))

        # ── Yaw (left-right turn) ─────────────────────────────────────────────
        # Idea: in a frontal face the nose sits at the midpoint between the eyes
        # horizontally.  Horizontal displacement from midpoint (normalised by IED/2)
        # maps to yaw angle.
        mid_x = eye_center[0]
        nose_offset_x = (nose[0] - mid_x) / (ied / 2.0 + 1e-6)
        # arcsin maps [-1,1] → [-90°,90°]; clip to avoid domain errors
        yaw = float(np.degrees(np.arcsin(np.clip(nose_offset_x, -0.95, 0.95))))

        # ── Pitch (up-down tilt) ──────────────────────────────────────────────
        # Use two independent ratios and average to reduce landmark noise:
        #  R1: vertical distance eye_center→nose / IED   (frontal ≈ 0.65)
        #  R2: vertical distance nose→mouth_center / IED (frontal ≈ 0.55)
        #
        # For pitch estimation we look at *deviation* of R1 from its frontal
        # reference, normalised.
        eye_nose_vert = float(nose[1] - eye_center[1])
        nose_mouth_vert = float(mouth_center[1] - nose[1])

        R1 = eye_nose_vert  / (ied + 1e-6)   # frontal reference ≈ 0.65
        R2 = nose_mouth_vert / (ied + 1e-6)  # frontal reference ≈ 0.55

        REF_R1, REF_R2 = 0.65, 0.55

        # Deviation from frontal for each ratio; map ±0.5 → ±45°
        p1 = (R1 - REF_R1) / 0.5 * 45.0
        p2 = (REF_R2 - R2) / 0.5 * 45.0   # R2 shrinks when looking up

        pitch = float(np.clip((p1 + p2) / 2.0, -90.0, 90.0))

        # ── Symmetry score ────────────────────────────────────────────────────
        # Measure: how asymmetric is the face horizontally around its mid-axis?
        # We compare left-eye→nose distance vs right-eye→nose distance (both
        # normalised by IED).  Perfect symmetry = ratio of 1.0.
        nose_to_le = abs(nose[0] - le[0]) / (ied + 1e-6)
        nose_to_re = abs(nose[0] - re[0]) / (ied + 1e-6)
        if (nose_to_le + nose_to_re) > 1e-6:
            asym = abs(nose_to_le - nose_to_re) / (nose_to_le + nose_to_re)
        else:
            asym = 0.0
        # asym = 0 → score 100; asym = 1 → score 0
        symmetry = float(np.clip((1.0 - asym) * 100.0, 0.0, 100.0))

        return yaw, pitch, roll, symmetry

    # ------------------------------------------------------------------
    # Pose → score
    # ------------------------------------------------------------------

    def _pose_score(self, yaw: float, pitch: float, roll: float) -> float:
        """Convert pose angles to a 0–100 score.

        Uses a soft cosine-based decay so small deviations barely penalise
        while large deviations ramp down smoothly rather than clipping hard.
        """
        # Soft penalty: 1 at angle=0, drops to 0 at max_angle
        max_yaw   = max(self.config.max_yaw_angle,   1.0)
        max_pitch = max(self.config.max_pitch_angle, 1.0)
        max_roll  = 45.0   # fixed; extreme roll (~>45°) is always problematic

        def _decay(angle: float, max_angle: float) -> float:
            # cos²(angle / max_angle * π/2) — smooth 1→0 as angle→max_angle
            ratio = min(abs(angle) / max_angle, 1.0)
            return float(np.cos(ratio * np.pi / 2) ** 2)

        yaw_s   = _decay(yaw,   max_yaw)
        pitch_s = _decay(pitch, max_pitch)
        roll_s  = _decay(roll,  max_roll)

        # Weight: yaw matters most for profile views, pitch next, roll least
        combined = 0.50 * yaw_s + 0.35 * pitch_s + 0.15 * roll_s
        return float(combined * 100.0)
