"""End-to-end face recognition pipeline — detection → alignment → quality → embedding → matching."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from src.alignment.aligner import FaceAligner
from src.alignment.quality import QualityAssessor, QualityReport
from src.config import SystemConfig, get_config
from src.detection.detector import FaceDetector
from src.detection.models import DetectedFace, DetectionResult
from src.recognition.embedder import FaceEmbedder
from src.recognition.matcher import GalleryMatcher, MatchResult

logger = logging.getLogger(__name__)


@dataclass
class RecognizedFace:
    """A fully processed face with detection, quality, embedding, and match info."""

    detected_face: DetectedFace
    aligned_crop: np.ndarray | None = None
    quality: QualityReport | None = None
    embedding: np.ndarray | None = None
    match: MatchResult | None = None
    identity_name: str = "Unknown"
    identity_id: str | None = None
    similarity: float = 0.0

    def to_dict(self) -> dict:
        return {
            "bbox": list(self.detected_face.bbox),
            "confidence": round(self.detected_face.confidence, 4),
            "identity_name": self.identity_name,
            "identity_id": self.identity_id,
            "similarity": round(self.similarity, 4),
            "quality": self.quality.to_dict() if self.quality else None,
        }


@dataclass
class PipelineResult:
    """Full pipeline result for a single frame."""

    recognized_faces: list[RecognizedFace] = field(default_factory=list)
    detection_result: DetectionResult | None = None
    total_time_ms: float = 0.0
    detection_time_ms: float = 0.0
    alignment_time_ms: float = 0.0
    embedding_time_ms: float = 0.0
    matching_time_ms: float = 0.0

    @property
    def num_detected(self) -> int:
        return self.detection_result.num_faces if self.detection_result else 0

    @property
    def num_recognized(self) -> int:
        return sum(1 for f in self.recognized_faces if f.identity_id is not None)

    def to_dict(self) -> dict:
        return {
            "num_detected": self.num_detected,
            "num_recognized": self.num_recognized,
            "total_time_ms": round(self.total_time_ms, 2),
            "timing": {
                "detection_ms": round(self.detection_time_ms, 2),
                "alignment_ms": round(self.alignment_time_ms, 2),
                "embedding_ms": round(self.embedding_time_ms, 2),
                "matching_ms": round(self.matching_time_ms, 2),
            },
            "faces": [f.to_dict() for f in self.recognized_faces],
        }


class FaceRecognitionPipeline:
    """Complete face recognition pipeline.

    Orchestrates: detection → alignment → quality gate → embedding → gallery matching.
    """

    def __init__(self, config: SystemConfig | None = None) -> None:
        self.config = config or get_config()
        self.detector = FaceDetector(self.config.detection)
        self.aligner = FaceAligner(self.config.alignment)
        self.quality_assessor = QualityAssessor(self.config.alignment)
        self.embedder = FaceEmbedder(self.config.recognition)
        self.gallery = GalleryMatcher(self.config.recognition)
        logger.info("FaceRecognitionPipeline initialized")

    def process_frame(
        self,
        image: np.ndarray,
        skip_quality: bool = False,
    ) -> PipelineResult:
        """Process a single frame through the full pipeline.

        Args:
            image: Input image as numpy array (BGR from OpenCV or RGB).
            skip_quality: If True, skip quality gate (process all detected faces).

        Returns:
            PipelineResult with all recognized faces and timing info.
        """
        total_start = time.perf_counter()
        result = PipelineResult()

        # Ensure RGB
        if image.ndim == 2:
            rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif image.shape[2] == 3:
            rgb = image  # Assume RGB; caller should convert if BGR
        else:
            rgb = image

        # --- Detection ---
        det_start = time.perf_counter()
        detection = self.detector.detect(Image.fromarray(rgb))
        result.detection_result = detection
        result.detection_time_ms = (time.perf_counter() - det_start) * 1000

        if detection.num_faces == 0:
            result.total_time_ms = (time.perf_counter() - total_start) * 1000
            return result

        # --- Alignment ---
        align_start = time.perf_counter()
        aligned_faces: list[tuple[DetectedFace, np.ndarray]] = []
        for face in detection.faces:
            aligned = self.aligner.align(rgb, face)
            if aligned is not None:
                aligned_faces.append((face, aligned))
        result.alignment_time_ms = (time.perf_counter() - align_start) * 1000

        # --- Quality Gate ---
        quality_passed: list[tuple[DetectedFace, np.ndarray, QualityReport]] = []
        for face, crop in aligned_faces:
            qr = self.quality_assessor.assess(crop, face)
            if skip_quality or qr.passed:
                quality_passed.append((face, crop, qr))
            else:
                result.recognized_faces.append(
                    RecognizedFace(
                        detected_face=face,
                        aligned_crop=crop,
                        quality=qr,
                    )
                )

        if not quality_passed:
            result.total_time_ms = (time.perf_counter() - total_start) * 1000
            return result

        # --- Embedding ---
        emb_start = time.perf_counter()
        crops = [crop for _, crop, _ in quality_passed]
        embeddings = self.embedder.embed_batch(crops)
        result.embedding_time_ms = (time.perf_counter() - emb_start) * 1000

        # --- Matching ---
        match_start = time.perf_counter()
        for i, (face, crop, qr) in enumerate(quality_passed):
            emb = embeddings[i]
            match = self.gallery.identify(emb)

            recognized = RecognizedFace(
                detected_face=face,
                aligned_crop=crop,
                quality=qr,
                embedding=emb,
                match=match,
            )
            if match is not None:
                recognized.identity_name = match.identity_name
                recognized.identity_id = match.identity_id
                recognized.similarity = match.similarity

            result.recognized_faces.append(recognized)
        result.matching_time_ms = (time.perf_counter() - match_start) * 1000

        result.total_time_ms = (time.perf_counter() - total_start) * 1000
        logger.info(
            "Pipeline: %d detected, %d recognized in %.1fms",
            result.num_detected,
            result.num_recognized,
            result.total_time_ms,
        )
        return result

    def enroll(
        self,
        identity_id: str,
        identity_name: str,
        images: list[np.ndarray],
    ) -> dict:
        """Enroll a new identity with one or more face images.

        Args:
            identity_id: Unique ID for the person.
            identity_name: Human-readable name.
            images: List of RGB numpy arrays containing the person's face.

        Returns:
            Enrollment summary dict.
        """
        total_embeddings = 0
        failed_images = 0

        for img in images:
            detection = self.detector.detect(Image.fromarray(img))
            if detection.num_faces == 0:
                failed_images += 1
                continue

            # Use the highest-confidence face
            best_face = detection.faces[0]
            aligned = self.aligner.align(img, best_face)
            if aligned is None:
                failed_images += 1
                continue

            embedding = self.embedder.embed(aligned)
            self.gallery.add(identity_id, identity_name, embedding)
            total_embeddings += 1

        return {
            "identity_id": identity_id,
            "identity_name": identity_name,
            "images_processed": len(images),
            "embeddings_created": total_embeddings,
            "failed_images": failed_images,
            "gallery_size": self.gallery.size,
        }

    def annotate_frame(
        self,
        frame: np.ndarray,
        result: PipelineResult,
    ) -> np.ndarray:
        """Draw bounding boxes and labels on a frame.

        Args:
            frame: Original frame (RGB numpy array).
            result: PipelineResult from process_frame.

        Returns:
            Annotated frame copy.
        """
        annotated = frame.copy()
        for face in result.recognized_faces:
            x1, y1, x2, y2 = [int(v) for v in face.detected_face.bbox]

            if face.identity_id:
                color = (0, 200, 0)  # Green for recognized
                label = f"{face.identity_name} ({face.similarity:.2f})"
            else:
                color = (200, 0, 0)  # Red for unknown
                label = "Unknown"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            # Label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return annotated
