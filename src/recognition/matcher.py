"""Gallery matching using FAISS — efficient similarity search for face embeddings."""

from __future__ import annotations

import logging
import pickle
import threading
from dataclasses import dataclass, field
from pathlib import Path

import faiss
import numpy as np

from src.config import RecognitionConfig, get_config

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Single match result from gallery search."""

    identity_id: str
    identity_name: str
    distance: float
    similarity: float

    def to_dict(self) -> dict:
        return {
            "identity_id": self.identity_id,
            "identity_name": self.identity_name,
            "distance": round(float(self.distance), 4),
            "similarity": round(float(self.similarity), 4),
        }


class GalleryMatcher:
    """FAISS-based gallery for face embedding matching.

    Supports add, remove, search, save, and load operations.
    Thread-safe for concurrent access.
    """

    def __init__(self, config: RecognitionConfig | None = None) -> None:
        self.config = config or get_config().recognition
        self._dim = self.config.embedding_dim
        self._lock = threading.Lock()

        # Identity metadata: maps internal index → (identity_id, identity_name)
        self._identities: list[tuple[str, str]] = []
        # All embeddings stored as (N, dim) for rebuild
        self._embeddings: list[np.ndarray] = []

        # FAISS index — using inner product (cosine sim on L2-normed vectors)
        self._index = faiss.IndexFlatIP(self._dim)

        logger.info(
            "GalleryMatcher initialized (dim=%d, threshold=%.2f)",
            self._dim,
            self.config.similarity_threshold,
        )

    @property
    def size(self) -> int:
        """Number of embeddings in the gallery."""
        return self._index.ntotal

    @property
    def identity_count(self) -> int:
        """Number of unique identities in the gallery."""
        return len(set(iid for iid, _ in self._identities))

    def add(
        self,
        identity_id: str,
        identity_name: str,
        embedding: np.ndarray,
    ) -> None:
        """Add a single embedding for an identity.

        Args:
            identity_id: Unique identifier for the person.
            identity_name: Human-readable name.
            embedding: L2-normalized 512-d embedding.
        """
        emb = embedding.reshape(1, -1).astype(np.float32)
        with self._lock:
            self._index.add(emb)
            self._identities.append((identity_id, identity_name))
            self._embeddings.append(embedding.copy())
        logger.debug("Added embedding for %s (%s). Gallery size: %d", identity_name, identity_id, self.size)

    def add_batch(
        self,
        identity_id: str,
        identity_name: str,
        embeddings: np.ndarray,
    ) -> None:
        """Add multiple embeddings for the same identity."""
        embs = embeddings.reshape(-1, self._dim).astype(np.float32)
        with self._lock:
            self._index.add(embs)
            for i in range(len(embs)):
                self._identities.append((identity_id, identity_name))
                self._embeddings.append(embs[i].copy())

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int | None = None,
    ) -> list[MatchResult]:
        """Search gallery for the closest matches to a query embedding.

        Args:
            query_embedding: L2-normalized 512-d query vector.
            top_k: Number of top matches to return (default from config).

        Returns:
            List of MatchResult sorted by similarity (descending).
        """
        if self.size == 0:
            return []

        k = min(top_k or self.config.top_k, self.size)
        query = query_embedding.reshape(1, -1).astype(np.float32)

        with self._lock:
            similarities, indices = self._index.search(query, k)

        results: list[MatchResult] = []
        for i in range(k):
            idx = int(indices[0][i])
            sim = float(similarities[0][i])
            if idx < 0:
                continue

            iid, name = self._identities[idx]
            distance = 1.0 - sim  # cosine distance
            results.append(
                MatchResult(
                    identity_id=iid,
                    identity_name=name,
                    distance=distance,
                    similarity=sim,
                )
            )

        return results

    def identify(
        self,
        query_embedding: np.ndarray,
    ) -> MatchResult | None:
        """Identify the best match above the similarity threshold.

        Returns the top match if its similarity exceeds the configured threshold,
        otherwise returns None (unknown identity).
        """
        matches = self.search(query_embedding, top_k=1)
        if not matches:
            return None
        if matches[0].similarity >= self.config.similarity_threshold:
            return matches[0]
        return None

    def remove_identity(self, identity_id: str) -> int:
        """Remove all embeddings for a given identity and rebuild the index.

        Returns the number of embeddings removed.
        """
        with self._lock:
            keep_indices = [
                i
                for i, (iid, _) in enumerate(self._identities)
                if iid != identity_id
            ]
            removed = len(self._identities) - len(keep_indices)

            if removed == 0:
                return 0

            self._identities = [self._identities[i] for i in keep_indices]
            self._embeddings = [self._embeddings[i] for i in keep_indices]

            # Rebuild FAISS index
            self._index = faiss.IndexFlatIP(self._dim)
            if self._embeddings:
                emb_array = np.stack(self._embeddings).astype(np.float32)
                self._index.add(emb_array)

        logger.info("Removed %d embeddings for identity %s", removed, identity_id)
        return removed

    def clear(self) -> None:
        """Remove all entries from the gallery."""
        with self._lock:
            self._index = faiss.IndexFlatIP(self._dim)
            self._identities.clear()
            self._embeddings.clear()
        logger.info("Gallery cleared")

    def save(self, path: str | Path) -> None:
        """Save gallery to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = {
                "identities": self._identities,
                "embeddings": [e.tolist() for e in self._embeddings],
                "dim": self._dim,
            }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info("Gallery saved to %s (%d entries)", path, self.size)

    def load(self, path: str | Path) -> None:
        """Load gallery from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Gallery file not found: {path}")

        with open(path, "rb") as f:
            data = pickle.load(f)  # noqa: S301

        with self._lock:
            self._identities = data["identities"]
            self._embeddings = [
                np.array(e, dtype=np.float32) for e in data["embeddings"]
            ]
            self._dim = data["dim"]
            self._index = faiss.IndexFlatIP(self._dim)
            if self._embeddings:
                emb_array = np.stack(self._embeddings).astype(np.float32)
                self._index.add(emb_array)

        logger.info("Gallery loaded from %s (%d entries)", path, self.size)

    def get_all_identities(self) -> list[dict]:
        """Return a list of unique identities with their embedding counts."""
        counts: dict[str, dict] = {}
        for iid, name in self._identities:
            if iid not in counts:
                counts[iid] = {"identity_id": iid, "name": name, "num_embeddings": 0}
            counts[iid]["num_embeddings"] += 1
        return list(counts.values())
