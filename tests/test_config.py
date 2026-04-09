"""Tests for the configuration module."""

from __future__ import annotations

from src.config import (
    AlignmentConfig,
    DetectionConfig,
    RecognitionConfig,
    SystemConfig,
    get_config,
    set_config,
)


class TestDetectionConfig:
    def test_defaults(self):
        c = DetectionConfig()
        assert c.model_name == "mtcnn"
        assert c.min_face_size == 40
        assert c.confidence_threshold == 0.90
        assert c.device == "cpu"
        assert c.keep_all is True

    def test_custom_values(self):
        c = DetectionConfig(min_face_size=20, confidence_threshold=0.5)
        assert c.min_face_size == 20
        assert c.confidence_threshold == 0.5


class TestAlignmentConfig:
    def test_defaults(self):
        c = AlignmentConfig()
        assert c.output_size == (160, 160)
        assert c.min_blur_score == 50.0

    def test_custom_size(self):
        c = AlignmentConfig(output_size=(112, 112))
        assert c.output_size == (112, 112)


class TestRecognitionConfig:
    def test_defaults(self):
        c = RecognitionConfig()
        assert c.embedding_dim == 512
        assert c.pretrained == "vggface2"

    def test_threshold(self):
        c = RecognitionConfig(similarity_threshold=0.8)
        assert c.similarity_threshold == 0.8


class TestSystemConfig:
    def test_creation(self):
        cfg = SystemConfig()
        assert cfg.detection.model_name == "mtcnn"
        assert cfg.recognition.embedding_dim == 512

    def test_from_env(self):
        cfg = SystemConfig.from_env()
        assert cfg.detection.device == "cpu"

    def test_directories_created(self, tmp_path):
        cfg = SystemConfig(
            data_dir=tmp_path / "data",
            gallery_dir=tmp_path / "gallery",
            logs_dir=tmp_path / "logs",
            models_dir=tmp_path / "models",
        )
        assert cfg.data_dir.exists()
        assert cfg.gallery_dir.exists()
        assert cfg.logs_dir.exists()
        assert cfg.models_dir.exists()


class TestConfigSingleton:
    def test_get_set_config(self):
        cfg = SystemConfig()
        set_config(cfg)
        retrieved = get_config()
        assert retrieved is cfg
