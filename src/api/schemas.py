"""Pydantic request/response schemas for the Face Recognition API."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# --- Detection ---

class DetectionFace(BaseModel):
    bbox: list[float] = Field(description="Bounding box [x1, y1, x2, y2]")
    confidence: float
    landmarks: list[list[float]] | None = None
    width: float
    height: float


class DetectionResponse(BaseModel):
    num_faces: int
    frame_size: list[int]
    inference_time_ms: float
    faces: list[DetectionFace]


# --- Recognition ---

class RecognizedFaceSchema(BaseModel):
    bbox: list[float]
    confidence: float
    identity_name: str
    identity_id: str | None
    similarity: float
    quality: dict[str, Any] | None = None
    top_candidate: dict[str, Any] | None = None  # closest gallery hit even below threshold


class RecognitionResponse(BaseModel):
    num_detected: int
    num_recognized: int
    total_time_ms: float
    timing: dict[str, float]
    faces: list[RecognizedFaceSchema]


# --- Enrollment ---

class EnrollRequest(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    metadata: dict[str, Any] | None = None


class EnrollResponse(BaseModel):
    identity_id: str
    identity_name: str
    images_processed: int
    embeddings_created: int
    embeddings_per_image: float = 0.0
    augmentation_enabled: bool = False
    failed_images: int
    avg_enrollment_quality: float = 0.0
    gallery_size: int


# --- Identity ---

class IdentitySchema(BaseModel):
    id: str
    name: str
    num_embeddings: int
    created_at: datetime | None = None
    updated_at: datetime | None = None
    is_active: bool = True


class IdentityListResponse(BaseModel):
    identities: list[IdentitySchema]
    total: int


class IdentityUpdateRequest(BaseModel):
    name: str | None = None


# --- Health ---

class HealthResponse(BaseModel):
    status: str
    version: str
    gallery_size: int
    identity_count: int
    device: str


# --- Config ---

class SystemConfigResponse(BaseModel):
    detection: dict[str, Any]
    alignment: dict[str, Any]
    recognition: dict[str, Any]


# --- Events ---

class RecognitionEventSchema(BaseModel):
    id: str
    identity_id: str | None
    identity_name: str | None
    similarity: float | None
    recognized: bool
    timestamp: datetime | None
