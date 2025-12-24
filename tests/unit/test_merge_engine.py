"""Unit tests for MergeEngine."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from recaller.models.note import Note, NoteStatus
from recaller.services.merge_engine import MergeEngine, MergeProposal, MergeResult
from recaller.services.similarity_engine import SimilarityPair


def make_note(
    note_id: int,
    title: str,
    category: str = "Test",
    source: str = "",
    content: str = "",
    embedding: np.ndarray = None,
) -> Note:
    """Helper to create a note with specified properties."""
    note = Note(
        id=note_id,
        notion_page_id=f"page-{note_id}",
        title=title,
        category=category,
        source=source,
        content=content,
    )
    note.embedding = embedding
    return note


@pytest.fixture
def mock_ollama():
    """Create a mocked OllamaService."""
    mock = MagicMock()
    mock.generate_combined_title.return_value = "Combined Title"
    mock.generate_combined_title_for_group.return_value = "Group Title"
    return mock


@pytest.fixture
def merge_engine(mock_ollama):
    """Create MergeEngine with mocked Ollama service."""
    return MergeEngine(ollama_service=mock_ollama)


class TestMergeProposal:
    """Tests for MergeProposal dataclass."""

    def test_primary_note_returns_first(self):
        """Test primary_note returns first note."""
        notes = [make_note(1, "First"), make_note(2, "Second")]
        proposal = MergeProposal(notes=notes, similarity_scores=[0.9])

        assert proposal.primary_note.title == "First"

    def test_secondary_notes_returns_rest(self):
        """Test secondary_notes returns all but first."""
        notes = [make_note(1, "First"), make_note(2, "Second"), make_note(3, "Third")]
        proposal = MergeProposal(notes=notes, similarity_scores=[0.9, 0.85])

        assert len(proposal.secondary_notes) == 2
        assert proposal.secondary_notes[0].title == "Second"
        assert proposal.secondary_notes[1].title == "Third"


class TestMergeResult:
    """Tests for MergeResult dataclass."""

    def test_str_representation(self):
        """Test string representation."""
        merged_note = make_note(1, "Merged Title")
        original_notes = [make_note(1, "Note A"), make_note(2, "Note B")]

        result = MergeResult(
            merged_note=merged_note,
            original_notes=original_notes,
            combined_title="Merged Title",
            similarity_scores=[0.9],
        )

        assert "2 notes" in str(result)
        assert "Merged Title" in str(result)


class TestCreateProposalsFromPairs:
    """Tests for creating merge proposals from similarity pairs."""

    def test_empty_pairs_returns_empty(self, merge_engine):
        """Test with empty pairs list."""
        result = merge_engine.create_proposals_from_pairs([])
        assert result == []

    def test_single_pair_creates_single_proposal(self, merge_engine):
        """Test single pair creates one proposal."""
        note_a = make_note(1, "Note A")
        note_b = make_note(2, "Note B")

        pairs = [SimilarityPair(note_a=note_a, note_b=note_b, similarity=0.9)]

        result = merge_engine.create_proposals_from_pairs(pairs)

        assert len(result) == 1
        assert len(result[0].notes) == 2
        assert result[0].similarity_scores == [0.9]

    def test_transitive_pairs_grouped(self, merge_engine):
        """Test that transitive pairs are grouped together."""
        note_a = make_note(1, "Note A")
        note_b = make_note(2, "Note B")
        note_c = make_note(3, "Note C")

        # A-B and B-C should result in one group of A, B, C
        pairs = [
            SimilarityPair(note_a=note_a, note_b=note_b, similarity=0.9),
            SimilarityPair(note_a=note_b, note_b=note_c, similarity=0.85),
        ]

        result = merge_engine.create_proposals_from_pairs(pairs)

        assert len(result) == 1
        assert len(result[0].notes) == 3

    def test_separate_pairs_create_separate_proposals(self, merge_engine):
        """Test that unrelated pairs create separate proposals."""
        note_a = make_note(1, "Note A")
        note_b = make_note(2, "Note B")
        note_c = make_note(3, "Note C")
        note_d = make_note(4, "Note D")

        # A-B and C-D are separate groups
        pairs = [
            SimilarityPair(note_a=note_a, note_b=note_b, similarity=0.9),
            SimilarityPair(note_a=note_c, note_b=note_d, similarity=0.85),
        ]

        result = merge_engine.create_proposals_from_pairs(pairs)

        assert len(result) == 2
        assert all(len(p.notes) == 2 for p in result)


class TestGenerateTitleForProposal:
    """Tests for title generation."""

    def test_generates_title_for_proposal(self, merge_engine, mock_ollama):
        """Test title generation calls Ollama service."""
        notes = [make_note(1, "First Title"), make_note(2, "Second Title")]
        proposal = MergeProposal(notes=notes, similarity_scores=[0.9])

        result = merge_engine.generate_title_for_proposal(proposal)

        mock_ollama.generate_combined_title_for_group.assert_called_once_with(
            ["First Title", "Second Title"]
        )
        assert result == "Group Title"


class TestExecuteMerge:
    """Tests for merge execution."""

    def test_execute_merge_creates_merged_note(self, merge_engine):
        """Test that execute_merge creates a merged note."""
        note_a = make_note(1, "Note A", content="Content A", source="Source A")
        note_b = make_note(2, "Note B", content="Content B", source="Source B")
        proposal = MergeProposal(notes=[note_a, note_b], similarity_scores=[0.9])

        result = merge_engine.execute_merge(proposal, "Final Title")

        assert isinstance(result, MergeResult)
        assert result.merged_note.title == "Final Title"
        assert result.merged_note.is_merge_parent is True

    def test_execute_merge_marks_secondary_as_merged(self, merge_engine):
        """Test that secondary notes are marked as merged."""
        note_a = make_note(1, "Note A")
        note_b = make_note(2, "Note B")
        note_c = make_note(3, "Note C")
        proposal = MergeProposal(
            notes=[note_a, note_b, note_c], similarity_scores=[0.9, 0.85]
        )

        result = merge_engine.execute_merge(proposal, "Final Title")

        # Secondary notes should be marked as merged
        for note in result.original_notes[1:]:
            assert note.status == NoteStatus.MERGED
            assert note.merge_group_id == note_a.id

    def test_execute_merge_combines_content(self, merge_engine):
        """Test that content is combined."""
        note_a = make_note(1, "Note A", content="Content A\n\nParagraph 1")
        note_b = make_note(2, "Note B", content="Content B\n\nParagraph 2")
        proposal = MergeProposal(notes=[note_a, note_b], similarity_scores=[0.9])

        result = merge_engine.execute_merge(proposal, "Final Title")

        assert "Content A" in result.merged_note.content
        assert "Content B" in result.merged_note.content
        assert "Paragraph 1" in result.merged_note.content
        assert "Paragraph 2" in result.merged_note.content

    def test_execute_merge_combines_sources(self, merge_engine):
        """Test that sources are combined."""
        note_a = make_note(1, "Note A", source="Wikipedia")
        note_b = make_note(2, "Note B", source="Stack Overflow")
        proposal = MergeProposal(notes=[note_a, note_b], similarity_scores=[0.9])

        result = merge_engine.execute_merge(proposal, "Final Title")

        assert "Wikipedia" in result.merged_note.source
        assert "Stack Overflow" in result.merged_note.source


class TestMergeContent:
    """Tests for content merging."""

    def test_merge_content_single_note(self, merge_engine):
        """Test content merge with single note."""
        note = make_note(1, "Note", content="Single content")

        result = merge_engine._merge_content([note])

        assert result == "Single content"

    def test_merge_content_deduplicates_paragraphs(self, merge_engine):
        """Test that duplicate paragraphs are removed."""
        note_a = make_note(1, "Note A", content="Same paragraph\n\nUnique A")
        note_b = make_note(2, "Note B", content="Same paragraph\n\nUnique B")

        result = merge_engine._merge_content([note_a, note_b])

        # Should have unique paragraphs only (case-insensitive dedup)
        assert result.count("Same paragraph") == 1
        assert "Unique A" in result
        assert "Unique B" in result

    def test_merge_content_preserves_order(self, merge_engine):
        """Test that content order is preserved."""
        note_a = make_note(1, "Note A", content="First")
        note_b = make_note(2, "Note B", content="Second")

        result = merge_engine._merge_content([note_a, note_b])

        # First should come before Second
        assert result.index("First") < result.index("Second")


class TestMergeSources:
    """Tests for source merging."""

    def test_merge_sources_empty(self, merge_engine):
        """Test merge sources with no sources."""
        note_a = make_note(1, "Note A", source="")
        note_b = make_note(2, "Note B", source="")

        result = merge_engine._merge_sources([note_a, note_b])

        assert result == ""

    def test_merge_sources_single_source(self, merge_engine):
        """Test merge sources with single source."""
        note_a = make_note(1, "Note A", source="Wikipedia")
        note_b = make_note(2, "Note B", source="")

        result = merge_engine._merge_sources([note_a, note_b])

        assert result == "Wikipedia"

    def test_merge_sources_deduplicates(self, merge_engine):
        """Test that duplicate sources are removed (case-insensitive)."""
        note_a = make_note(1, "Note A", source="Wikipedia")
        note_b = make_note(2, "Note B", source="wikipedia")

        result = merge_engine._merge_sources([note_a, note_b])

        assert result == "Wikipedia"

    def test_merge_sources_combines_multiple(self, merge_engine):
        """Test combining multiple sources."""
        note_a = make_note(1, "Note A", source="Wikipedia")
        note_b = make_note(2, "Note B", source="Stack Overflow")

        result = merge_engine._merge_sources([note_a, note_b])

        assert "Wikipedia" in result
        assert "Stack Overflow" in result
        assert ", " in result


class TestPreviewMerge:
    """Tests for merge preview."""

    def test_preview_merge_returns_info(self, merge_engine):
        """Test that preview returns expected information."""
        note_a = make_note(
            1, "Note A", category="Tech", source="Wikipedia", content="Content A"
        )
        note_b = make_note(
            2, "Note B", category="Tech", source="Stack Overflow", content="Content B"
        )
        proposal = MergeProposal(notes=[note_a, note_b], similarity_scores=[0.9])

        result = merge_engine.preview_merge(proposal)

        assert result["note_count"] == 2
        assert result["titles"] == ["Note A", "Note B"]
        assert result["category"] == "Tech"
        assert "Wikipedia" in result["merged_source"]
        assert result["avg_similarity"] == 0.9

    def test_preview_merge_empty_scores(self, merge_engine):
        """Test preview with no similarity scores."""
        note_a = make_note(1, "Note A")
        proposal = MergeProposal(notes=[note_a], similarity_scores=[])

        result = merge_engine.preview_merge(proposal)

        assert result["avg_similarity"] == 0.0
