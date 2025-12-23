"""SQLAlchemy database schema for Recaller."""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)

from recaller.models.note import NoteStatus
from recaller.models.flashcard import FlashcardType, ExportStatus


class Base(DeclarativeBase):
    """Base class for all database models."""

    pass


class NoteRecord(Base):
    """Database record for a note."""

    __tablename__ = "notes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    notion_page_id: Mapped[str] = mapped_column(String(36), unique=True, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    category: Mapped[str] = mapped_column(String(255), nullable=True)
    source: Mapped[str] = mapped_column(Text, nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
    notion_last_edited: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    status: Mapped[str] = mapped_column(
        Enum(NoteStatus), default=NoteStatus.NEW, nullable=False
    )
    embedding: Mapped[Optional[bytes]] = mapped_column(LargeBinary, nullable=True)

    merge_group_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("notes.id"), nullable=True
    )
    is_merge_parent: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Relationships
    flashcards: Mapped[list["FlashcardRecord"]] = relationship(
        "FlashcardRecord", back_populates="note", cascade="all, delete-orphan"
    )
    merged_notes: Mapped[list["NoteRecord"]] = relationship(
        "NoteRecord", backref="merge_parent", remote_side=[id]
    )

    __table_args__ = (
        Index("idx_notes_status", "status"),
        Index("idx_notes_merge_group", "merge_group_id"),
        Index("idx_notes_notion_page_id", "notion_page_id"),
    )


class FlashcardRecord(Base):
    """Database record for a flashcard."""

    __tablename__ = "flashcards"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    note_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("notes.id", ondelete="CASCADE"), nullable=False
    )

    front: Mapped[str] = mapped_column(Text, nullable=False)
    back: Mapped[str] = mapped_column(Text, nullable=False)
    card_type: Mapped[str] = mapped_column(
        Enum(FlashcardType), default=FlashcardType.BASIC, nullable=False
    )
    tags: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON array
    deck_name: Mapped[str] = mapped_column(
        String(255), default="Recaller::Weekly", nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    export_status: Mapped[str] = mapped_column(
        Enum(ExportStatus), default=ExportStatus.PENDING, nullable=False
    )
    anki_note_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    export_batch_id: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    # Relationships
    note: Mapped["NoteRecord"] = relationship("NoteRecord", back_populates="flashcards")

    __table_args__ = (
        Index("idx_flashcards_export_status", "export_status"),
        Index("idx_flashcards_note_id", "note_id"),
    )


class MergeHistoryRecord(Base):
    """Audit trail for merge operations."""

    __tablename__ = "merge_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    parent_note_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("notes.id"), nullable=False
    )
    child_note_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("notes.id"), nullable=False
    )
    similarity_score: Mapped[float] = mapped_column(Float, nullable=False)
    merged_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )


class ExportBatchRecord(Base):
    """Track weekly export batches."""

    __tablename__ = "export_batches"

    id: Mapped[str] = mapped_column(String(50), primary_key=True)  # e.g., "2025-01-15"
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )
    notes_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    flashcards_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="pending", nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


def get_engine(database_url: str):
    """Create database engine."""
    return create_engine(database_url, echo=False)


def get_session_factory(engine) -> sessionmaker[Session]:
    """Create session factory."""
    return sessionmaker(bind=engine, expire_on_commit=False)


def init_database(database_url: str) -> sessionmaker[Session]:
    """Initialize database and return session factory."""
    engine = get_engine(database_url)
    Base.metadata.create_all(engine)
    return get_session_factory(engine)
