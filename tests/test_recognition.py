"""Tests for the face embedding and gallery matching modules."""

from __future__ import annotations

import numpy as np
import pytest

from src.config import RecognitionConfig
from src.recognition.embedder import FaceEmbedder
from src.recognition.matcher import GalleryMatcher, MatchResult


@pytest.fixture(scope="module")
def embedder():
    return FaceEmbedder(RecognitionConfig(device="cpu"))


@pytest.fixture
def gallery():
    return GalleryMatcher(RecognitionConfig(device="cpu", similarity_threshold=0.4))


class TestFaceEmbedder:
    def test_embed_single(self, embedder):
        """Single face embedding should be 512-d and L2-normalized."""
        face = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        emb = embedder.embed(face)
        assert emb.shape == (512,)
        assert emb.dtype == np.float32
        # Check L2 normalization
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 0.01

    def test_embed_batch(self, embedder):
        """Batch embedding should handle multiple faces."""
        faces = [np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8) for _ in range(3)]
        embs = embedder.embed_batch(faces)
        assert embs.shape == (3, 512)
        for i in range(3):
            norm = np.linalg.norm(embs[i])
            assert abs(norm - 1.0) < 0.01

    def test_embed_batch_empty(self, embedder):
        """Empty batch should return empty array."""
        embs = embedder.embed_batch([])
        assert embs.shape == (0, 512)

    def test_same_face_similar_embedding(self, embedder):
        """Same input should produce identical embedding."""
        face = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        emb1 = embedder.embed(face)
        emb2 = embedder.embed(face)
        similarity = np.dot(emb1, emb2)
        assert similarity > 0.99

    def test_different_faces_different_embeddings(self, embedder):
        """Different random inputs should produce different embeddings."""
        face1 = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        face2 = np.full((160, 160, 3), 128, dtype=np.uint8)
        emb1 = embedder.embed(face1)
        emb2 = embedder.embed(face2)
        # Should not be identical
        assert not np.allclose(emb1, emb2, atol=0.01)


class TestGalleryMatcher:
    def test_add_and_search(self, gallery):
        """Add an identity and search for it."""
        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)
        gallery.add("id1", "Alice", emb)
        assert gallery.size == 1

        results = gallery.search(emb)
        assert len(results) >= 1
        assert results[0].identity_name == "Alice"
        assert results[0].similarity > 0.9

    def test_identify_above_threshold(self, gallery):
        """Identify should return a match above the threshold."""
        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)
        gallery.add("id2", "Bob", emb)

        match = gallery.identify(emb)
        assert match is not None
        assert match.identity_name == "Bob"

    def test_identify_below_threshold(self):
        """Identify should return None for unknown faces."""
        matcher = GalleryMatcher(RecognitionConfig(similarity_threshold=0.99))
        emb1 = np.random.randn(512).astype(np.float32)
        emb1 /= np.linalg.norm(emb1)
        matcher.add("id1", "Alice", emb1)

        emb2 = np.random.randn(512).astype(np.float32)
        emb2 /= np.linalg.norm(emb2)
        match = matcher.identify(emb2)
        # Random vectors in 512-d are almost orthogonal, so similarity should be low
        assert match is None

    def test_empty_gallery_search(self, gallery):
        """Search on empty gallery returns empty list."""
        empty_gallery = GalleryMatcher()
        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)
        results = empty_gallery.search(emb)
        assert results == []

    def test_remove_identity(self, gallery):
        """Remove identity should remove all its embeddings."""
        emb1 = np.random.randn(512).astype(np.float32)
        emb1 /= np.linalg.norm(emb1)
        emb2 = np.random.randn(512).astype(np.float32)
        emb2 /= np.linalg.norm(emb2)

        gallery.clear()
        gallery.add("rm1", "Charlie", emb1)
        gallery.add("rm1", "Charlie", emb2)
        assert gallery.size == 2

        removed = gallery.remove_identity("rm1")
        assert removed == 2
        assert gallery.size == 0

    def test_clear_gallery(self, gallery):
        """Clear should empty the gallery."""
        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)
        gallery.add("x", "X", emb)
        gallery.clear()
        assert gallery.size == 0
        assert gallery.identity_count == 0

    def test_save_and_load(self, gallery, tmp_path):
        """Save and load should preserve gallery contents."""
        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)
        gallery.clear()
        gallery.add("save1", "Dave", emb)

        path = tmp_path / "test_gallery.pkl"
        gallery.save(path)

        new_gallery = GalleryMatcher()
        new_gallery.load(path)
        assert new_gallery.size == 1

        match = new_gallery.identify(emb)
        assert match is not None
        assert match.identity_name == "Dave"

    def test_load_nonexistent(self, gallery, tmp_path):
        """Load from nonexistent file should raise."""
        with pytest.raises(FileNotFoundError):
            gallery.load(tmp_path / "nonexistent.pkl")

    def test_get_all_identities(self, gallery):
        """get_all_identities returns unique identity summaries."""
        gallery.clear()
        emb = np.random.randn(512).astype(np.float32)
        emb /= np.linalg.norm(emb)
        gallery.add("i1", "Alice", emb)
        gallery.add("i1", "Alice", emb)
        gallery.add("i2", "Bob", emb)

        identities = gallery.get_all_identities()
        assert len(identities) == 2
        alice = next(i for i in identities if i["name"] == "Alice")
        assert alice["num_embeddings"] == 2

    def test_match_result_to_dict(self):
        """MatchResult serialization works."""
        mr = MatchResult("id1", "Alice", 0.15, 0.85)
        d = mr.to_dict()
        assert d["identity_name"] == "Alice"
        assert d["similarity"] == 0.85
