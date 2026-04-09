"""Face embedding generation using InceptionResnetV1 (FaceNet architecture)."""

from __future__ import annotations

import logging
import time

import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from torchvision import transforms

from src.config import RecognitionConfig, get_config

logger = logging.getLogger(__name__)


class FaceEmbedder:
    """Generate 512-dimensional face embeddings from aligned face crops.

    Uses InceptionResnetV1 pretrained on VGGFace2 (or CASIA-WebFace).
    Produces L2-normalized 512-d vectors for cosine/euclidean similarity.
    """

    def __init__(self, config: RecognitionConfig | None = None) -> None:
        self.config = config or get_config().recognition
        self._device = torch.device(self.config.device)
        self._model = InceptionResnetV1(
            pretrained=self.config.pretrained,
            classify=False,
            device=self._device,
        ).eval()

        # Standard preprocessing for InceptionResnetV1
        self._transform = transforms.Compose(
            [
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                ),
            ]
        )

        logger.info(
            "FaceEmbedder initialized (%s, pretrained=%s, device=%s)",
            self.config.model_name,
            self.config.pretrained,
            self.config.device,
        )

    def embed(self, aligned_face: np.ndarray) -> np.ndarray:
        """Generate embedding for a single aligned face crop.

        Args:
            aligned_face: Aligned face image as RGB numpy array (H, W, 3).

        Returns:
            512-dimensional L2-normalized embedding as numpy array.
        """
        pil = Image.fromarray(aligned_face.astype(np.uint8))
        tensor = self._transform(pil).unsqueeze(0).to(self._device)

        with torch.no_grad():
            embedding = self._model(tensor)

        emb = embedding.cpu().numpy().flatten()
        # L2 normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        return emb.astype(np.float32)

    def embed_batch(self, aligned_faces: list[np.ndarray]) -> np.ndarray:
        """Generate embeddings for a batch of aligned face crops.

        Args:
            aligned_faces: List of aligned face images (RGB numpy arrays).

        Returns:
            Array of shape (N, 512) with L2-normalized embeddings.
        """
        if not aligned_faces:
            return np.empty((0, self.config.embedding_dim), dtype=np.float32)

        start = time.perf_counter()

        tensors = []
        for face in aligned_faces:
            pil = Image.fromarray(face.astype(np.uint8))
            tensors.append(self._transform(pil))

        batch = torch.stack(tensors).to(self._device)

        with torch.no_grad():
            embeddings = self._model(batch)

        embs = embeddings.cpu().numpy()
        # L2 normalize each row
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        embs = embs / norms

        elapsed = (time.perf_counter() - start) * 1000
        logger.debug(
            "Generated %d embeddings in %.1fms (%.1fms/face)",
            len(aligned_faces),
            elapsed,
            elapsed / len(aligned_faces),
        )

        return embs.astype(np.float32)
