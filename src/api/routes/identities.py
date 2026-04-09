"""Identity management API routes."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    IdentityListResponse,
    IdentitySchema,
    IdentityUpdateRequest,
)

router = APIRouter(prefix="/identities", tags=["identities"])


@router.get("/", response_model=IdentityListResponse)
async def list_identities(offset: int = 0, limit: int = 100):
    """List all registered identities."""
    from src.api.app import get_pipeline, get_repository

    repo = get_repository()
    pipeline = get_pipeline()

    if repo:
        identities = await repo.list_identities(offset=offset, limit=limit)
        return {
            "identities": [
                IdentitySchema(
                    id=i.id,
                    name=i.name,
                    num_embeddings=i.num_embeddings,
                    created_at=i.created_at,
                    updated_at=i.updated_at,
                    is_active=i.is_active,
                )
                for i in identities
            ],
            "total": len(identities),
        }

    # Fallback: return from gallery
    gallery_identities = pipeline.gallery.get_all_identities()
    return {
        "identities": [
            IdentitySchema(
                id=gi["identity_id"],
                name=gi["name"],
                num_embeddings=gi["num_embeddings"],
                is_active=True,
            )
            for gi in gallery_identities
        ],
        "total": len(gallery_identities),
    }


@router.get("/{identity_id}", response_model=IdentitySchema)
async def get_identity(identity_id: str):
    """Get a specific identity by ID."""
    from src.api.app import get_repository

    repo = get_repository()
    if not repo:
        raise HTTPException(status_code=503, detail="Database not available")

    identity = await repo.get_identity(identity_id)
    if identity is None:
        raise HTTPException(status_code=404, detail="Identity not found")

    return IdentitySchema(
        id=identity.id,
        name=identity.name,
        num_embeddings=identity.num_embeddings,
        created_at=identity.created_at,
        updated_at=identity.updated_at,
        is_active=identity.is_active,
    )


@router.patch("/{identity_id}", response_model=IdentitySchema)
async def update_identity(identity_id: str, body: IdentityUpdateRequest):
    """Update an identity's name."""
    from src.api.app import get_repository

    repo = get_repository()
    if not repo:
        raise HTTPException(status_code=503, detail="Database not available")

    identity = await repo.update_identity(identity_id, name=body.name)
    if identity is None:
        raise HTTPException(status_code=404, detail="Identity not found")

    return IdentitySchema(
        id=identity.id,
        name=identity.name,
        num_embeddings=identity.num_embeddings,
        created_at=identity.created_at,
        updated_at=identity.updated_at,
        is_active=identity.is_active,
    )


@router.delete("/{identity_id}")
async def delete_identity(identity_id: str):
    """Soft-delete an identity and remove from gallery."""
    from src.api.app import get_pipeline, get_repository

    repo = get_repository()
    pipeline = get_pipeline()

    removed = pipeline.gallery.remove_identity(identity_id)

    if repo:
        deleted = await repo.delete_identity(identity_id)
        if not deleted and removed == 0:
            raise HTTPException(status_code=404, detail="Identity not found")

    return {
        "detail": "Identity deleted",
        "identity_id": identity_id,
        "embeddings_removed": removed,
    }
