"""Repository for database CRUD operations."""

import hashlib
import json
from datetime import datetime
from typing import Optional

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from recaller.database.schema import (
    ExportBatchRecord,
    FlashcardRecord,
    MergeHistoryRecord,
    NoteRecord,
    init_database,
)
from recaller.models.flashcard import ExportStatus, Flashcard, FlashcardType
from recaller.models.note import Note, NoteStatus


class Repository:
    """Repository for managing notes and flashcards in the database."""

    def __init__(self, database_url: str):
        """Initialize repository with database connection."""
        self.session_factory = init_database(database_url)

    def _get_session(self) -> Session:
        """Get a new database session."""
        return self.session_factory()

    # ==================== Note Operations ====================

    def add_note(self, note: Note) -> Note:
        """Add a new note to the database."""
        with self._get_session() as session:
            record = NoteRecord(
                notion_page_id=note.notion_page_id,
                title=note.title,
                category=note.category,
                source=note.source,
                content=note.content,
                content_hash=self._compute_hash(note.title, note.content),
                notion_last_edited=note.notion_last_edited,
                status=note.status,
                embedding=self._serialize_embedding(note.embedding),
                merge_group_id=note.merge_group_id,
                is_merge_parent=note.is_merge_parent,
            )
            session.add(record)
            session.commit()
            note.id = record.id
            note.content_hash = record.content_hash
            return note

    def get_note_by_id(self, note_id: int) -> Optional[Note]:
        """Get a note by its database ID."""
        with self._get_session() as session:
            record = session.get(NoteRecord, note_id)
            if record:
                return self._record_to_note(record)
            return None

    def get_note_by_notion_id(self, notion_page_id: str) -> Optional[Note]:
        """Get a note by its Notion page ID."""
        with self._get_session() as session:
            stmt = select(NoteRecord).where(NoteRecord.notion_page_id == notion_page_id)
            record = session.scalars(stmt).first()
            if record:
                return self._record_to_note(record)
            return None

    def get_notes_by_status(self, status: NoteStatus) -> list[Note]:
        """Get all notes with a specific status."""
        with self._get_session() as session:
            stmt = select(NoteRecord).where(NoteRecord.status == status)
            records = session.scalars(stmt).all()
            return [self._record_to_note(r) for r in records]

    def get_all_notes(self, include_archived: bool = False) -> list[Note]:
        """Get all notes, optionally including archived ones."""
        with self._get_session() as session:
            if include_archived:
                stmt = select(NoteRecord)
            else:
                stmt = select(NoteRecord).where(NoteRecord.status != NoteStatus.ARCHIVED)
            records = session.scalars(stmt).all()
            return [self._record_to_note(r) for r in records]

    def update_note(self, note: Note) -> Note:
        """Update an existing note."""
        with self._get_session() as session:
            record = session.get(NoteRecord, note.id)
            if record:
                record.title = note.title
                record.category = note.category
                record.source = note.source
                record.content = note.content
                record.content_hash = self._compute_hash(note.title, note.content)
                record.notion_last_edited = note.notion_last_edited
                record.status = note.status
                record.embedding = self._serialize_embedding(note.embedding)
                record.merge_group_id = note.merge_group_id
                record.is_merge_parent = note.is_merge_parent
                record.updated_at = datetime.utcnow()
                session.commit()
                note.content_hash = record.content_hash
                note.updated_at = record.updated_at
            return note

    def update_note_embedding(self, note_id: int, embedding: np.ndarray) -> None:
        """Update just the embedding for a note."""
        with self._get_session() as session:
            record = session.get(NoteRecord, note_id)
            if record:
                record.embedding = self._serialize_embedding(embedding)
                record.updated_at = datetime.utcnow()
                session.commit()

    def update_note_status(self, note_id: int, status: NoteStatus) -> None:
        """Update just the status for a note."""
        with self._get_session() as session:
            record = session.get(NoteRecord, note_id)
            if record:
                record.status = status
                record.updated_at = datetime.utcnow()
                session.commit()

    # ==================== Flashcard Operations ====================

    def add_flashcard(self, flashcard: Flashcard) -> Flashcard:
        """Add a new flashcard to the database."""
        with self._get_session() as session:
            record = FlashcardRecord(
                note_id=flashcard.note_id,
                front=flashcard.front,
                back=flashcard.back,
                card_type=flashcard.card_type,
                tags=json.dumps(flashcard.tags),
                deck_name=flashcard.deck_name,
                export_status=flashcard.export_status,
                anki_note_id=flashcard.anki_note_id,
                export_batch_id=flashcard.export_batch_id,
            )
            session.add(record)
            session.commit()
            flashcard.id = record.id
            return flashcard

    def add_flashcards(self, flashcards: list[Flashcard]) -> list[Flashcard]:
        """Add multiple flashcards in a single transaction."""
        with self._get_session() as session:
            for flashcard in flashcards:
                record = FlashcardRecord(
                    note_id=flashcard.note_id,
                    front=flashcard.front,
                    back=flashcard.back,
                    card_type=flashcard.card_type,
                    tags=json.dumps(flashcard.tags),
                    deck_name=flashcard.deck_name,
                    export_status=flashcard.export_status,
                )
                session.add(record)
                session.flush()  # Get the ID
                flashcard.id = record.id
            session.commit()
        return flashcards

    def get_flashcards_by_status(self, status: ExportStatus) -> list[Flashcard]:
        """Get all flashcards with a specific export status."""
        with self._get_session() as session:
            stmt = select(FlashcardRecord).where(FlashcardRecord.export_status == status)
            records = session.scalars(stmt).all()
            return [self._record_to_flashcard(r) for r in records]

    def get_flashcards_for_note(self, note_id: int) -> list[Flashcard]:
        """Get all flashcards for a specific note."""
        with self._get_session() as session:
            stmt = select(FlashcardRecord).where(FlashcardRecord.note_id == note_id)
            records = session.scalars(stmt).all()
            return [self._record_to_flashcard(r) for r in records]

    def update_flashcard_export(
        self,
        flashcard_id: int,
        status: ExportStatus,
        anki_note_id: Optional[int] = None,
        batch_id: Optional[str] = None,
    ) -> None:
        """Update flashcard export status."""
        with self._get_session() as session:
            record = session.get(FlashcardRecord, flashcard_id)
            if record:
                record.export_status = status
                if anki_note_id is not None:
                    record.anki_note_id = anki_note_id
                if batch_id is not None:
                    record.export_batch_id = batch_id
                session.commit()

    # ==================== Merge Operations ====================

    def record_merge(
        self, parent_note_id: int, child_note_id: int, similarity_score: float
    ) -> None:
        """Record a merge operation in the history."""
        with self._get_session() as session:
            record = MergeHistoryRecord(
                parent_note_id=parent_note_id,
                child_note_id=child_note_id,
                similarity_score=similarity_score,
            )
            session.add(record)
            session.commit()

    # ==================== Export Batch Operations ====================

    def create_export_batch(self, batch_id: str) -> None:
        """Create a new export batch record."""
        with self._get_session() as session:
            record = ExportBatchRecord(id=batch_id)
            session.add(record)
            session.commit()

    def update_export_batch(
        self,
        batch_id: str,
        notes_count: int,
        flashcards_count: int,
        status: str,
        error_message: Optional[str] = None,
    ) -> None:
        """Update an export batch record."""
        with self._get_session() as session:
            record = session.get(ExportBatchRecord, batch_id)
            if record:
                record.notes_count = notes_count
                record.flashcards_count = flashcards_count
                record.status = status
                record.error_message = error_message
                session.commit()

    # ==================== Statistics ====================

    def get_stats(self) -> dict:
        """Get database statistics."""
        with self._get_session() as session:
            total_notes = session.query(NoteRecord).count()
            new_notes = (
                session.query(NoteRecord)
                .filter(NoteRecord.status == NoteStatus.NEW)
                .count()
            )
            processed_notes = (
                session.query(NoteRecord)
                .filter(NoteRecord.status == NoteStatus.PROCESSED)
                .count()
            )
            merged_notes = (
                session.query(NoteRecord)
                .filter(NoteRecord.status == NoteStatus.MERGED)
                .count()
            )
            total_flashcards = session.query(FlashcardRecord).count()
            pending_flashcards = (
                session.query(FlashcardRecord)
                .filter(FlashcardRecord.export_status == ExportStatus.PENDING)
                .count()
            )
            exported_flashcards = (
                session.query(FlashcardRecord)
                .filter(FlashcardRecord.export_status == ExportStatus.EXPORTED)
                .count()
            )

            return {
                "total_notes": total_notes,
                "new_notes": new_notes,
                "processed_notes": processed_notes,
                "merged_notes": merged_notes,
                "total_flashcards": total_flashcards,
                "pending_flashcards": pending_flashcards,
                "exported_flashcards": exported_flashcards,
            }

    # ==================== Helper Methods ====================

    def _record_to_note(self, record: NoteRecord) -> Note:
        """Convert database record to Note model."""
        return Note(
            id=record.id,
            notion_page_id=record.notion_page_id,
            title=record.title,
            category=record.category or "",
            source=record.source or "",
            content=record.content,
            content_hash=record.content_hash,
            created_at=record.created_at,
            updated_at=record.updated_at,
            notion_last_edited=record.notion_last_edited,
            status=NoteStatus(
                record.status.value if hasattr(record.status, "value") else record.status
            ),
            embedding=self._deserialize_embedding(record.embedding),
            merge_group_id=record.merge_group_id,
            is_merge_parent=record.is_merge_parent,
        )

    def _record_to_flashcard(self, record: FlashcardRecord) -> Flashcard:
        """Convert database record to Flashcard model."""
        return Flashcard(
            id=record.id,
            note_id=record.note_id,
            front=record.front,
            back=record.back,
            card_type=FlashcardType(
                record.card_type.value
                if hasattr(record.card_type, "value")
                else record.card_type
            ),
            tags=json.loads(record.tags) if record.tags else [],
            deck_name=record.deck_name,
            created_at=record.created_at,
            export_status=ExportStatus(
                record.export_status.value
                if hasattr(record.export_status, "value")
                else record.export_status
            ),
            anki_note_id=record.anki_note_id,
            export_batch_id=record.export_batch_id,
        )

    @staticmethod
    def _compute_hash(title: str, content: str) -> str:
        """Compute SHA256 hash of title and content for change detection."""
        combined = f"{title}\n{content}"
        return hashlib.sha256(combined.encode()).hexdigest()

    @staticmethod
    def _serialize_embedding(embedding: Optional[np.ndarray]) -> Optional[bytes]:
        """Serialize numpy array to bytes for storage."""
        if embedding is None:
            return None
        return embedding.tobytes()

    @staticmethod
    def _deserialize_embedding(data: Optional[bytes]) -> Optional[np.ndarray]:
        """Deserialize bytes to numpy array."""
        if data is None:
            return None
        return np.frombuffer(data, dtype=np.float32)
