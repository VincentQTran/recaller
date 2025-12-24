"""Flashcard model for Recaller."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class FlashcardType(str, Enum):
    """Type of flashcard."""

    BASIC = "basic"  # Front/Back question-answer
    CLOZE = "cloze"  # Fill in the blank


class ExportStatus(str, Enum):
    """Export status for a flashcard."""

    PENDING = "pending"  # Not yet exported to Anki
    EXPORTED = "exported"  # Successfully exported
    FAILED = "failed"  # Export failed


@dataclass
class Flashcard:
    """Represents an Anki flashcard generated from a note."""

    # Card content
    front: str  # Question or prompt
    back: str  # Answer
    card_type: FlashcardType = FlashcardType.BASIC

    # Relationship to source note
    note_id: int = 0

    # Anki metadata
    tags: list[str] = field(default_factory=list)
    deck_name: str = ""  # Set dynamically to current date (MM-DD-YYYY)

    # Local database metadata
    id: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Export tracking
    export_status: ExportStatus = ExportStatus.PENDING
    anki_note_id: Optional[int] = None  # Anki's internal note ID
    export_batch_id: Optional[str] = None  # Weekly batch identifier

    def __post_init__(self) -> None:
        """Validate and convert fields after initialization."""
        if isinstance(self.card_type, str):
            self.card_type = FlashcardType(self.card_type)
        if isinstance(self.export_status, str):
            self.export_status = ExportStatus(self.export_status)

    def to_anki_note(self) -> dict:
        """Convert to AnkiConnect note format."""
        return {
            "deckName": self.deck_name,
            "modelName": "Basic",
            "fields": {
                "Front": self.front,
                "Back": self.back,
            },
            "tags": self.tags,
        }
