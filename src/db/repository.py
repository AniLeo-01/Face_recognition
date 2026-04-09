"""Database repository for identity CRUD operations."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.config import DatabaseConfig, get_config
from src.db.models import Base, EnrollmentRecord, Identity, RecognitionEvent

logger = logging.getLogger(__name__)


class IdentityRepository:
    """Async repository for managing identities and recognition events."""

    def __init__(self, config: DatabaseConfig | None = None) -> None:
        self.config = config or get_config().database
        self._engine = create_async_engine(
            self.config.url, echo=self.config.echo
        )
        self._session_factory = async_sessionmaker(
            self._engine, expire_on_commit=False
        )

    async def init_db(self) -> None:
        """Create all tables."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created")

    async def close(self) -> None:
        await self._engine.dispose()

    def _session(self) -> AsyncSession:
        return self._session_factory()

    # --- Identity CRUD ---

    async def create_identity(
        self, name: str, metadata: dict | None = None
    ) -> Identity:
        identity = Identity(
            id=str(uuid.uuid4()),
            name=name,
            metadata_json=json.dumps(metadata) if metadata else None,
        )
        async with self._session() as session:
            session.add(identity)
            await session.commit()
            await session.refresh(identity)
        logger.info("Created identity: %s (%s)", name, identity.id)
        return identity

    async def get_identity(self, identity_id: str) -> Identity | None:
        async with self._session() as session:
            return await session.get(Identity, identity_id)

    async def get_identity_by_name(self, name: str) -> Identity | None:
        async with self._session() as session:
            result = await session.execute(
                select(Identity).where(Identity.name == name, Identity.is_active.is_(True))
            )
            return result.scalar_one_or_none()

    async def list_identities(
        self, offset: int = 0, limit: int = 100
    ) -> list[Identity]:
        async with self._session() as session:
            result = await session.execute(
                select(Identity)
                .where(Identity.is_active.is_(True))
                .order_by(Identity.created_at.desc())
                .offset(offset)
                .limit(limit)
            )
            return list(result.scalars().all())

    async def update_identity(
        self,
        identity_id: str,
        name: str | None = None,
        num_embeddings: int | None = None,
    ) -> Identity | None:
        async with self._session() as session:
            identity = await session.get(Identity, identity_id)
            if identity is None:
                return None
            if name is not None:
                identity.name = name
            if num_embeddings is not None:
                identity.num_embeddings = num_embeddings
            identity.updated_at = datetime.now(timezone.utc)
            await session.commit()
            await session.refresh(identity)
            return identity

    async def delete_identity(self, identity_id: str) -> bool:
        async with self._session() as session:
            identity = await session.get(Identity, identity_id)
            if identity is None:
                return False
            identity.is_active = False
            identity.updated_at = datetime.now(timezone.utc)
            await session.commit()
        logger.info("Deactivated identity: %s", identity_id)
        return True

    # --- Enrollment Records ---

    async def create_enrollment_record(
        self,
        identity_id: str,
        image_path: str | None = None,
        embedding_count: int = 1,
    ) -> EnrollmentRecord:
        record = EnrollmentRecord(
            id=str(uuid.uuid4()),
            identity_id=identity_id,
            image_path=image_path,
            embedding_count=embedding_count,
        )
        async with self._session() as session:
            session.add(record)
            await session.commit()
        return record

    # --- Recognition Events ---

    async def log_recognition_event(
        self,
        identity_id: str | None,
        identity_name: str | None,
        similarity: float | None,
        bbox: list[float] | None = None,
        frame_source: str | None = None,
        recognized: bool = False,
    ) -> RecognitionEvent:
        event = RecognitionEvent(
            id=str(uuid.uuid4()),
            identity_id=identity_id,
            identity_name=identity_name,
            similarity=similarity,
            bbox_json=json.dumps(bbox) if bbox else None,
            frame_source=frame_source,
            recognized=recognized,
        )
        async with self._session() as session:
            session.add(event)
            await session.commit()
        return event

    async def get_recent_events(
        self, limit: int = 50
    ) -> list[RecognitionEvent]:
        async with self._session() as session:
            result = await session.execute(
                select(RecognitionEvent)
                .order_by(RecognitionEvent.timestamp.desc())
                .limit(limit)
            )
            return list(result.scalars().all())
