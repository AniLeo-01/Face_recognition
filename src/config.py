"""Central configuration for the Face Recognition System."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class DetectionConfig:
    """Face detection configuration."""

    model_name: Literal["mtcnn"] = "mtcnn"
    min_face_size: int = 40
    confidence_threshold: float = 0.90
    device: str = "cpu"
    # MTCNN thresholds for three stages (P-Net, R-Net, O-Net)
    thresholds: list[float] = field(default_factory=lambda: [0.6, 0.7, 0.7])
    select_largest: bool = False
    keep_all: bool = True


@dataclass
class AlignmentConfig:
    """Face alignment and quality configuration.

    Quality is now scored on a continuous 0–100 scale (composite of blur,
    pose, face-size, and symmetry) rather than a set of hard per-metric limits.
    Faces are placed in one of three tiers:

        PASS     quality_score >= hard_quality_threshold  (default 55)
                 → embed + match with the standard similarity threshold
        MARGINAL quality_score >= soft_quality_threshold  (default 30)
                 → embed + match with a tightened similarity threshold
        FAIL     quality_score <  soft_quality_threshold
                 → skip embedding entirely

    Individual per-metric caps still exist to catch truly unusable angles/sizes,
    but they feed into the score rather than causing immediate rejection.

    All thresholds can be overridden via environment variables:
        QUALITY_HARD_THRESHOLD  (default 55)
        QUALITY_SOFT_THRESHOLD  (default 30)
        QUALITY_MAX_YAW         (default 60.0 degrees)
        QUALITY_MAX_PITCH       (default 45.0 degrees)
    """

    output_size: tuple[int, int] = (160, 160)
    min_face_size: int = 30

    # ── Composite quality thresholds ──────────────────────────────────────────
    hard_quality_threshold: float = field(
        default_factory=lambda: float(os.getenv("QUALITY_HARD_THRESHOLD", "55"))
    )
    soft_quality_threshold: float = field(
        default_factory=lambda: float(os.getenv("QUALITY_SOFT_THRESHOLD", "30"))
    )

    # ── Per-metric caps (feed into pose_score; do NOT cause direct rejection) ─
    max_yaw_angle: float = field(
        default_factory=lambda: float(os.getenv("QUALITY_MAX_YAW", "60.0"))
    )
    max_pitch_angle: float = field(
        default_factory=lambda: float(os.getenv("QUALITY_MAX_PITCH", "45.0"))
    )


@dataclass
class RecognitionConfig:
    """Face recognition / embedding configuration."""

    model_name: Literal["inception_resnet_v1"] = "inception_resnet_v1"
    pretrained: Literal["vggface2", "casia-webface"] = "vggface2"
    embedding_dim: int = 512
    device: str = "cpu"
    # Matching
    similarity_threshold: float = 0.55          # threshold for PASS-tier faces (cross-domain: portrait→scene)
    marginal_similarity_boost: float = 0.04     # added to threshold for MARGINAL-tier faces
    top_k: int = 5
    # Enrollment augmentation — generates N synthetic variants per photo to
    # broaden gallery coverage (handles lighting, angle, and flip variation).
    # Set to 0 to disable; 5 is a good default (flip + 2 brightness + 2 rotation)
    enrollment_augmentation: bool = True
    enrollment_aug_brightness_steps: int = 2   # ±20%, ±40% → 4 extra embeddings
    enrollment_aug_rotation_deg: float = 10.0  # rotate ±N° → 2 extra embeddings
    # After matching, suppress duplicate detections of the same identity:
    # keep only the highest-similarity hit per identity_id per frame.
    deduplicate_identities: bool = True
    # FAISS index type
    faiss_index_type: Literal["flat", "ivf"] = "flat"
    faiss_nprobe: int = 10


@dataclass
class APIConfig:
    """API server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    cors_origins: list[str] = field(default_factory=lambda: ["*"])
    max_upload_size_mb: int = 50
    rate_limit_per_minute: int = 120


@dataclass
class DatabaseConfig:
    """Database configuration."""

    url: str = "sqlite+aiosqlite:///./face_recognition.db"
    echo: bool = False


@dataclass
class SystemConfig:
    """Top-level system configuration."""

    detection: DetectionConfig = field(default_factory=DetectionConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    api: APIConfig = field(default_factory=APIConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    # Paths
    data_dir: Path = Path("./data")
    gallery_dir: Path = Path("./data/gallery")
    logs_dir: Path = Path("./logs")
    models_dir: Path = Path("./models")

    def __post_init__(self) -> None:
        for d in [self.data_dir, self.gallery_dir, self.logs_dir, self.models_dir]:
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> SystemConfig:
        """Create config from environment variables with sensible defaults."""
        device = "cuda" if os.getenv("USE_GPU", "").lower() == "true" else "cpu"
        db_url = os.getenv(
            "DATABASE_URL", "sqlite+aiosqlite:///./face_recognition.db"
        )
        return cls(
            detection=DetectionConfig(device=device),
            recognition=RecognitionConfig(device=device),
            database=DatabaseConfig(url=db_url),
        )


# Singleton config
_config: SystemConfig | None = None


def get_config() -> SystemConfig:
    global _config
    if _config is None:
        _config = SystemConfig.from_env()
    return _config


def set_config(config: SystemConfig) -> None:
    global _config
    _config = config
