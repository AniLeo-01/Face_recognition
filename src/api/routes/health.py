"""Health and configuration API routes."""

from __future__ import annotations

import dataclasses

from fastapi import APIRouter

from src import __version__
from src.api.schemas import HealthResponse, SystemConfigResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check endpoint."""
    from src.api.app import get_pipeline

    pipeline = get_pipeline()
    return {
        "status": "healthy",
        "version": __version__,
        "gallery_size": pipeline.gallery.size,
        "identity_count": pipeline.gallery.identity_count,
        "device": pipeline.config.detection.device,
    }


@router.get("/config", response_model=SystemConfigResponse)
async def get_system_config():
    """Get current system configuration."""
    from src.api.app import get_pipeline

    pipeline = get_pipeline()
    cfg = pipeline.config
    return {
        "detection": dataclasses.asdict(cfg.detection),
        "alignment": dataclasses.asdict(cfg.alignment),
        "recognition": {
            k: v
            for k, v in dataclasses.asdict(cfg.recognition).items()
        },
    }


@router.get("/events")
async def get_recent_events(limit: int = 50):
    """Get recent recognition events (audit log)."""
    from src.api.app import get_repository

    repo = get_repository()
    if not repo:
        return {"events": [], "total": 0}

    events = await repo.get_recent_events(limit=limit)
    return {
        "events": [
            {
                "id": e.id,
                "identity_id": e.identity_id,
                "identity_name": e.identity_name,
                "similarity": e.similarity,
                "recognized": e.recognized,
                "timestamp": e.timestamp.isoformat() if e.timestamp else None,
            }
            for e in events
        ],
        "total": len(events),
    }
