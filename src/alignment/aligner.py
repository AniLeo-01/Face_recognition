"""Face alignment module — landmark-based affine transformation."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src.config import AlignmentConfig, get_config
from src.detection.models import DetectedFace

logger = logging.getLogger(__name__)

# Reference landmarks for a canonical 112x112 aligned face (ArcFace standard).
# These are the target positions for: left_eye, right_eye, nose, mouth_left, mouth_right.
ARCFACE_REF_LANDMARKS_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def _scale_landmarks(
    ref: np.ndarray, target_size: tuple[int, int]
) -> np.ndarray:
    """Scale reference landmarks from 112x112 to target output size."""
    sx = target_size[0] / 112.0
    sy = target_size[1] / 112.0
    scaled = ref.copy()
    scaled[:, 0] *= sx
    scaled[:, 1] *= sy
    return scaled


class FaceAligner:
    """Align detected faces using landmark-based affine transformation.

    Produces normalized, canonical face crops suitable for embedding generation.
    """

    def __init__(self, config: AlignmentConfig | None = None) -> None:
        self.config = config or get_config().alignment
        self._ref_landmarks = _scale_landmarks(
            ARCFACE_REF_LANDMARKS_112, self.config.output_size
        )
        logger.info(
            "FaceAligner initialized (output_size=%s)", self.config.output_size
        )

    def align(
        self,
        image: np.ndarray,
        face: DetectedFace,
    ) -> np.ndarray | None:
        """Align a single detected face from the source image.

        Args:
            image: Source image (RGB numpy array).
            face: DetectedFace with landmarks.

        Returns:
            Aligned face crop as RGB numpy array of shape (H, W, 3),
            or None if alignment fails (e.g., missing landmarks).
        """
        if face.landmarks is None or face.landmarks.shape != (5, 2):
            # Fallback: simple crop + resize without alignment
            return self._crop_and_resize(image, face)

        src_pts = face.landmarks.astype(np.float32)
        dst_pts = self._ref_landmarks

        # Estimate affine transformation (similarity transform)
        tform = cv2.estimateAffinePartial2D(src_pts, dst_pts)[0]
        if tform is None:
            logger.warning("Affine estimation failed, falling back to crop")
            return self._crop_and_resize(image, face)

        aligned = cv2.warpAffine(
            image,
            tform,
            self.config.output_size,
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return aligned

    def align_multiple(
        self,
        image: np.ndarray,
        faces: list[DetectedFace],
    ) -> list[np.ndarray]:
        """Align multiple detected faces from the same source image."""
        results = []
        for face in faces:
            aligned = self.align(image, face)
            if aligned is not None:
                results.append(aligned)
        return results

    def _crop_and_resize(
        self, image: np.ndarray, face: DetectedFace
    ) -> np.ndarray:
        """Fallback: crop bounding box and resize."""
        h, w = image.shape[:2]
        x1, y1, x2, y2 = face.bbox
        # Add margin
        margin_x = (x2 - x1) * 0.1
        margin_y = (y2 - y1) * 0.1
        x1 = max(0, int(x1 - margin_x))
        y1 = max(0, int(y1 - margin_y))
        x2 = min(w, int(x2 + margin_x))
        y2 = min(h, int(y2 + margin_y))

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros(
                (*self.config.output_size[::-1], 3), dtype=np.uint8
            )
        return cv2.resize(crop, self.config.output_size)
