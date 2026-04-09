"""Detection API routes."""

from __future__ import annotations

import io

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

from src.api.schemas import DetectionResponse

router = APIRouter(prefix="/detect", tags=["detection"])


@router.post("/", response_model=DetectionResponse)
async def detect_faces(file: UploadFile = File(...)):
    """Detect faces in an uploaded image.

    Returns bounding boxes, confidence scores, and landmarks for all detected faces.
    """
    from src.api.app import get_pipeline

    pipeline = get_pipeline()

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = np.array(pil_image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    result = pipeline.detector.detect(pil_image)
    return result.to_dict()
