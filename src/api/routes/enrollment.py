"""Enrollment API routes."""

from __future__ import annotations

import io
from typing import List

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

from src.api.schemas import EnrollResponse

router = APIRouter(prefix="/enroll", tags=["enrollment"])


@router.post("/", response_model=EnrollResponse)
async def enroll_identity(
    name: str = Form(...),
    files: List[UploadFile] = File(...),
):
    """Enroll a new identity with one or more face images.

    The system will detect the primary face in each image, generate embeddings,
    and add them to the recognition gallery.
    """
    from src.api.app import get_pipeline, get_repository

    pipeline = get_pipeline()
    repo = get_repository()

    if not files:
        raise HTTPException(status_code=400, detail="At least one image is required")

    # Create identity in DB
    identity_id = None
    if repo:
        try:
            existing = await repo.get_identity_by_name(name)
            if existing:
                identity_id = existing.id
            else:
                identity = await repo.create_identity(name=name)
                identity_id = identity.id
        except Exception:
            pass

    if identity_id is None:
        import uuid
        identity_id = str(uuid.uuid4())

    images: list[np.ndarray] = []
    for f in files:
        contents = await f.read()
        if not contents:
            continue
        try:
            pil = Image.open(io.BytesIO(contents)).convert("RGB")
            images.append(np.array(pil))
        except Exception:
            continue

    if not images:
        raise HTTPException(status_code=400, detail="No valid images provided")

    result = pipeline.enroll(identity_id, name, images)

    # Update DB
    if repo and result["embeddings_created"] > 0:
        try:
            await repo.update_identity(
                identity_id, num_embeddings=result["embeddings_created"]
            )
            await repo.create_enrollment_record(
                identity_id=identity_id,
                embedding_count=result["embeddings_created"],
            )
        except Exception:
            pass

    return result
