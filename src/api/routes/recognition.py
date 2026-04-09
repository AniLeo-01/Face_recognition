"""Recognition API routes."""

from __future__ import annotations

import io

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

from src.api.schemas import RecognitionResponse

router = APIRouter(prefix="/recognize", tags=["recognition"])


@router.post("/", response_model=RecognitionResponse)
async def recognize_faces(file: UploadFile = File(...)):
    """Recognize faces in an uploaded image.

    Runs the full pipeline: detection → alignment → quality → embedding → matching.
    Returns identified faces with similarity scores.
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

    result = pipeline.process_frame(image)

    # Log recognition events asynchronously
    from src.api.app import get_repository

    repo = get_repository()
    if repo:
        for face in result.recognized_faces:
            try:
                await repo.log_recognition_event(
                    identity_id=face.identity_id,
                    identity_name=face.identity_name,
                    similarity=face.similarity if face.similarity > 0 else None,
                    bbox=list(face.detected_face.bbox),
                    recognized=face.identity_id is not None,
                )
            except Exception:
                pass  # Don't fail the request for audit logging errors

    return result.to_dict()
