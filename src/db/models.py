"""SQLAlchemy ORM models for the face recognition system."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Identity(Base):
    """Registered identity (person) in the system."""

    __tablename__ = "identities"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    num_embeddings: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
    is_active: Mapped[bool] = mapped_column(default=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "num_embeddings": self.num_embeddings,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "is_active": self.is_active,
        }


class EnrollmentRecord(Base):
    """Record of an enrollment event (image processed for identity)."""

    __tablename__ = "enrollment_records"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    identity_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    image_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    embedding_count: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
    )


class RecognitionEvent(Base):
    """Audit log for recognition events."""

    __tablename__ = "recognition_events"

    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )
    identity_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True, index=True
    )
    identity_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    similarity: Mapped[float | None] = mapped_column(Float, nullable=True)
    bbox_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    frame_source: Mapped[str | None] = mapped_column(String(512), nullable=True)
    recognized: Mapped[bool] = mapped_column(default=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )
