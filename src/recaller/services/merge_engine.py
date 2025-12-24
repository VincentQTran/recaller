"""Merge engine for combining similar notes."""

from dataclasses import dataclass
from typing import Any, Optional

from recaller.models.note import Note, NoteStatus
from recaller.services.ollama_service import OllamaService
from recaller.services.similarity_engine import SimilarityPair


@dataclass
class MergeResult:
    """Result of a merge operation."""

    merged_note: Note
    original_notes: list[Note]
    combined_title: str
    similarity_scores: list[float]

    def __str__(self) -> str:
        return f"Merged {len(self.original_notes)} notes -> '{self.combined_title}'"


@dataclass
class MergeProposal:
    """A proposed merge for user confirmation."""

    notes: list[Note]
    similarity_scores: list[float]
    suggested_title: Optional[str] = None

    @property
    def primary_note(self) -> Note:
        """The first note in the group (used as merge parent)."""
        return self.notes[0]

    @property
    def secondary_notes(self) -> list[Note]:
        """All notes except the primary."""
        return self.notes[1:]


class MergeEngine:
    """Engine for merging similar notes.

    Handles the logic for:
    - Generating combined titles via Ollama
    - Merging content from multiple notes
    - Deduplicating paragraphs
    - Combining sources
    """

    def __init__(self, ollama_service: OllamaService):
        """Initialize the merge engine.

        Args:
            ollama_service: OllamaService instance for title generation
        """
        self.llm = ollama_service

    def create_proposals_from_pairs(
        self, pairs: list[SimilarityPair]
    ) -> list[MergeProposal]:
        """Convert similarity pairs into merge proposals.

        Groups connected notes together into single proposals.

        Args:
            pairs: List of similar note pairs

        Returns:
            List of MergeProposal objects
        """
        if not pairs:
            return []

        # Build adjacency graph
        adjacency: dict[str, set[str]] = {}
        note_lookup: dict[str, Note] = {}
        score_lookup: dict[tuple[str, str], float] = {}

        for pair in pairs:
            id_a = pair.note_a.notion_page_id
            id_b = pair.note_b.notion_page_id

            if id_a not in adjacency:
                adjacency[id_a] = set()
            if id_b not in adjacency:
                adjacency[id_b] = set()

            adjacency[id_a].add(id_b)
            adjacency[id_b].add(id_a)

            note_lookup[id_a] = pair.note_a
            note_lookup[id_b] = pair.note_b

            # Store score with sorted key for consistent lookup
            key = tuple(sorted([id_a, id_b]))
            score_lookup[key] = pair.similarity  # type: ignore

        # Find connected components using BFS
        visited: set[str] = set()
        proposals: list[MergeProposal] = []

        for start_id in adjacency:
            if start_id in visited:
                continue

            # BFS to find all connected notes
            group_ids: list[str] = []
            queue = [start_id]

            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue

                visited.add(current)
                group_ids.append(current)

                for neighbor in adjacency.get(current, []):
                    if neighbor not in visited:
                        queue.append(neighbor)

            # Convert to notes and collect scores
            notes = [note_lookup[nid] for nid in group_ids]
            scores = []
            for i, id_a in enumerate(group_ids):
                for id_b in group_ids[i + 1 :]:
                    key = tuple(sorted([id_a, id_b]))
                    if key in score_lookup:
                        scores.append(score_lookup[key])  # type: ignore

            proposals.append(MergeProposal(notes=notes, similarity_scores=scores))

        return proposals

    def generate_title_for_proposal(self, proposal: MergeProposal) -> str:
        """Generate a combined title for a merge proposal.

        Args:
            proposal: The merge proposal

        Returns:
            Generated combined title
        """
        titles = [note.title for note in proposal.notes]
        return self.llm.generate_combined_title_for_group(titles)

    def execute_merge(self, proposal: MergeProposal, final_title: str) -> MergeResult:
        """Execute a merge operation.

        Creates a new merged note from the proposal.

        Args:
            proposal: The approved merge proposal
            final_title: The final title to use (may be edited by user)

        Returns:
            MergeResult with the new merged note
        """
        primary = proposal.primary_note
        secondary = proposal.secondary_notes

        # Merge content - combine all content with headers
        merged_content = self._merge_content(proposal.notes)

        # Combine sources
        merged_source = self._merge_sources(proposal.notes)

        # Create the merged note (as a new note based on primary)
        merged_note = Note(
            notion_page_id=primary.notion_page_id,  # Keep primary's ID
            title=final_title,
            category=primary.category,  # Keep primary's category
            source=merged_source,
            content=merged_content,
            id=primary.id,  # Keep primary's DB ID
            status=NoteStatus.PROCESSED,
            embedding=primary.embedding,  # Keep primary's embedding
            is_merge_parent=True,
        )

        # Mark secondary notes for archival
        for note in secondary:
            note.status = NoteStatus.MERGED
            note.merge_group_id = primary.id

        return MergeResult(
            merged_note=merged_note,
            original_notes=proposal.notes,
            combined_title=final_title,
            similarity_scores=proposal.similarity_scores,
        )

    def _merge_content(self, notes: list[Note]) -> str:
        """Merge content from multiple notes.

        Combines content with deduplication of similar paragraphs.

        Args:
            notes: List of notes to merge

        Returns:
            Combined content string
        """
        if len(notes) == 1:
            return notes[0].content

        # Collect all unique paragraphs
        seen_paragraphs: set[str] = set()
        content_parts: list[str] = []

        for note in notes:
            paragraphs = note.content.split("\n\n")
            for para in paragraphs:
                para_stripped = para.strip()
                if not para_stripped:
                    continue

                # Simple deduplication: check if paragraph is already seen
                para_normalized = para_stripped.lower()
                if para_normalized not in seen_paragraphs:
                    seen_paragraphs.add(para_normalized)
                    content_parts.append(para_stripped)

        return "\n\n".join(content_parts)

    def _merge_sources(self, notes: list[Note]) -> str:
        """Merge sources from multiple notes.

        Combines unique sources with comma separation.

        Args:
            notes: List of notes to merge

        Returns:
            Combined source string
        """
        sources: list[str] = []
        seen: set[str] = set()

        for note in notes:
            if note.source and note.source.strip():
                source = note.source.strip()
                if source.lower() not in seen:
                    seen.add(source.lower())
                    sources.append(source)

        return ", ".join(sources) if sources else ""

    def preview_merge(self, proposal: MergeProposal) -> dict[str, Any]:
        """Generate a preview of what the merge would produce.

        Args:
            proposal: The merge proposal to preview

        Returns:
            Dictionary with preview information
        """
        merged_content = self._merge_content(proposal.notes)
        merged_source = self._merge_sources(proposal.notes)

        return {
            "note_count": len(proposal.notes),
            "titles": [n.title for n in proposal.notes],
            "category": proposal.primary_note.category,
            "merged_source": merged_source,
            "content_length": len(merged_content),
            "avg_similarity": (
                sum(proposal.similarity_scores) / len(proposal.similarity_scores)
                if proposal.similarity_scores
                else 0.0
            ),
        }
