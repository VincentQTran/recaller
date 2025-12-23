"""Note model for Recaller."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import numpy as np


class NoteStatus(str, Enum):
    """Status of a note in the processing pipeline."""

    NEW = "new"  # Just ingested from Notion
    PROCESSED = "processed"  # Embeddings generated, ready for flashcards
    MERGED = "merged"  # Has been merged into another note
    ARCHIVED = "archived"  # Superseded by a merged note


@dataclass
class Note:
    """Represents a note from Notion."""

    # Core fields from Notion
    notion_page_id: str
    title: str
    category: str
    source: str
    content: str

    # Local database metadata
    id: Optional[int] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    notion_last_edited: Optional[datetime] = None

    # Processing state
    status: NoteStatus = NoteStatus.NEW
    embedding: Optional[np.ndarray] = None
    content_hash: Optional[str] = None

    # Merge tracking
    merge_group_id: Optional[int] = None  # FK to parent note if merged
    is_merge_parent: bool = False  # True if this is a combined note

    def __post_init__(self) -> None:
        """Validate and convert fields after initialization."""
        if isinstance(self.status, str):
            self.status = NoteStatus(self.status)

    def __hash__(self) -> int:
        """Hash based on Notion page ID for set operations."""
        return hash(self.notion_page_id)

    def __eq__(self, other: object) -> bool:
        """Equality based on Notion page ID."""
        if not isinstance(other, Note):
            return NotImplemented
        return self.notion_page_id == other.notion_page_id
