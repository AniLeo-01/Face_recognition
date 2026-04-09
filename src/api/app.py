"""FastAPI application — main entry point for the Face Recognition API."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src import __version__
from src.api.routes import detection, enrollment, health, identities, recognition
from src.config import SystemConfig, get_config
from src.db.repository import IdentityRepository
from src.pipeline.processor import FaceRecognitionPipeline

logger = logging.getLogger(__name__)

# Module-level singletons
_pipeline: FaceRecognitionPipeline | None = None
_repository: IdentityRepository | None = None


def get_pipeline() -> FaceRecognitionPipeline:
    """Get the global pipeline singleton."""
    if _pipeline is None:
        raise RuntimeError("Pipeline not initialized. Start the application first.")
    return _pipeline


def get_repository() -> IdentityRepository | None:
    """Get the global repository singleton (may be None if DB is unavailable)."""
    return _repository


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize and teardown resources."""
    global _pipeline, _repository

    config = get_config()
    logger.info("Initializing Face Recognition System v%s", __version__)

    # Initialize pipeline
    _pipeline = FaceRecognitionPipeline(config)
    logger.info("Pipeline initialized (device=%s)", config.detection.device)

    # Initialize database
    try:
        _repository = IdentityRepository(config.database)
        await _repository.init_db()
        logger.info("Database initialized: %s", config.database.url)
    except Exception as e:
        logger.warning("Database initialization failed (running without DB): %s", e)
        _repository = None

    # Load gallery from disk if exists
    gallery_path = config.gallery_dir / "gallery.pkl"
    if gallery_path.exists():
        try:
            _pipeline.gallery.load(gallery_path)
            logger.info("Loaded gallery from %s", gallery_path)
        except Exception as e:
            logger.warning("Failed to load gallery: %s", e)

    yield

    # Shutdown: save gallery
    if _pipeline:
        try:
            config.gallery_dir.mkdir(parents=True, exist_ok=True)
            _pipeline.gallery.save(config.gallery_dir / "gallery.pkl")
            logger.info("Gallery saved on shutdown")
        except Exception as e:
            logger.warning("Failed to save gallery: %s", e)

    if _repository:
        await _repository.close()

    logger.info("Face Recognition System shut down")


def create_app(config: SystemConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if config:
        from src.config import set_config
        set_config(config)

    app = FastAPI(
        title="Face Detection & Recognition System",
        description=(
            "Production-grade face detection and recognition API. "
            "Supports multi-face detection, alignment, quality assessment, "
            "embedding generation, and gallery-based identification."
        ),
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    cfg = get_config()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    app.include_router(health.router)
    app.include_router(detection.router)
    app.include_router(recognition.router)
    app.include_router(enrollment.router)
    app.include_router(identities.router)

    return app


# Default app instance for uvicorn
app = create_app()
