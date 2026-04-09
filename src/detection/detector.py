"""Face detection module using MTCNN from facenet-pytorch."""

from __future__ import annotations

import logging
import time
from typing import Union

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN
from PIL import Image

from src.config import DetectionConfig, get_config
from src.detection.models import DetectedFace, DetectionResult

logger = logging.getLogger(__name__)


class FaceDetector:
    """Multi-face detector using MTCNN.

    Detects faces in images/frames and returns bounding boxes,
    confidence scores, and 5-point facial landmarks.
    """

    def __init__(self, config: DetectionConfig | None = None) -> None:
        self.config = config or get_config().detection
        self._device = torch.device(self.config.device)
        self._model = MTCNN(
            image_size=160,
            margin=14,
            min_face_size=self.config.min_face_size,
            thresholds=self.config.thresholds,
            factor=0.709,
            keep_all=self.config.keep_all,
            select_largest=self.config.select_largest,
            device=self._device,
        )
        logger.info(
            "FaceDetector initialized on %s (min_face=%d, threshold=%.2f)",
            self.config.device,
            self.config.min_face_size,
            self.config.confidence_threshold,
        )

    def detect(
        self,
        image: Union[np.ndarray, Image.Image],
    ) -> DetectionResult:
        """Detect faces in an image.

        Args:
            image: Input image as numpy array (BGR, from OpenCV) or PIL Image (RGB).

        Returns:
            DetectionResult with detected faces, bounding boxes, and landmarks.
        """
        # Convert to PIL if numpy
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
        else:
            pil_image = image.convert("RGB")
            image = np.array(pil_image)

        h, w = image.shape[:2]

        start = time.perf_counter()
        boxes, probs, landmarks = self._model.detect(pil_image, landmarks=True)
        elapsed_ms = (time.perf_counter() - start) * 1000

        faces: list[DetectedFace] = []
        if boxes is not None and probs is not None:
            for i in range(len(boxes)):
                conf = float(probs[i])
                if conf < self.config.confidence_threshold:
                    continue

                box = boxes[i]
                # Clamp to image bounds
                x1 = max(0.0, float(box[0]))
                y1 = max(0.0, float(box[1]))
                x2 = min(float(w), float(box[2]))
                y2 = min(float(h), float(box[3]))

                lm = None
                if landmarks is not None:
                    lm = np.array(landmarks[i], dtype=np.float32)

                faces.append(
                    DetectedFace(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        landmarks=lm,
                    )
                )

        # Sort by confidence descending
        faces.sort(key=lambda f: f.confidence, reverse=True)

        logger.debug(
            "Detected %d faces in %.1fms (frame %dx%d)",
            len(faces),
            elapsed_ms,
            w,
            h,
        )

        return DetectionResult(
            faces=faces,
            frame_width=w,
            frame_height=h,
            inference_time_ms=elapsed_ms,
        )

    def detect_from_path(self, image_path: str) -> DetectionResult:
        """Detect faces from an image file path."""
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        return self.detect(img)

    def detect_from_video_frame(
        self, frame: np.ndarray
    ) -> DetectionResult:
        """Detect faces from a video frame (BGR numpy array)."""
        return self.detect(frame)
