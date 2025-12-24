"""Unit tests for SimilarityEngine."""

import numpy as np
import pytest

from recaller.models.note import Note
from recaller.services.similarity_engine import SimilarityEngine, SimilarityPair


@pytest.fixture
def similarity_engine():
    """Create SimilarityEngine with default threshold."""
    return SimilarityEngine(threshold=0.78)


def make_note(
    note_id: int,
    title: str,
    category: str = "Test",
    embedding: np.ndarray = None,
) -> Note:
    """Helper to create a note with an embedding."""
    note = Note(
        id=note_id,
        notion_page_id=f"page-{note_id}",
        title=title,
        category=category,
        source="",
        content="",
    )
    note.embedding = embedding
    return note


class TestSimilarityEngineInit:
    """Tests for SimilarityEngine initialization."""

    def test_init_default_threshold(self):
        """Test default threshold value."""
        engine = SimilarityEngine()
        assert engine.threshold == 0.78

    def test_init_custom_threshold(self):
        """Test custom threshold value."""
        engine = SimilarityEngine(threshold=0.85)
        assert engine.threshold == 0.85

    def test_init_invalid_threshold_high(self):
        """Test that threshold > 1 raises error."""
        with pytest.raises(ValueError):
            SimilarityEngine(threshold=1.5)

    def test_init_invalid_threshold_low(self):
        """Test that threshold < 0 raises error."""
        with pytest.raises(ValueError):
            SimilarityEngine(threshold=-0.1)


class TestCosineSimilarity:
    """Tests for cosine similarity computation."""

    def test_identical_vectors(self, similarity_engine):
        """Test similarity of identical vectors is 1."""
        vec = np.array([1.0, 2.0, 3.0])
        result = similarity_engine.cosine_similarity(vec, vec)
        assert abs(result - 1.0) < 1e-6

    def test_orthogonal_vectors(self, similarity_engine):
        """Test similarity of orthogonal vectors is 0."""
        vec_a = np.array([1.0, 0.0])
        vec_b = np.array([0.0, 1.0])
        result = similarity_engine.cosine_similarity(vec_a, vec_b)
        assert abs(result) < 1e-6

    def test_opposite_vectors(self, similarity_engine):
        """Test similarity of opposite vectors is -1."""
        vec_a = np.array([1.0, 2.0, 3.0])
        vec_b = np.array([-1.0, -2.0, -3.0])
        result = similarity_engine.cosine_similarity(vec_a, vec_b)
        assert abs(result - (-1.0)) < 1e-6

    def test_zero_vector(self, similarity_engine):
        """Test similarity with zero vector is 0."""
        vec_a = np.array([1.0, 2.0, 3.0])
        vec_b = np.array([0.0, 0.0, 0.0])
        result = similarity_engine.cosine_similarity(vec_a, vec_b)
        assert result == 0.0


class TestFindSimilarPairs:
    """Tests for finding similar note pairs."""

    def test_find_similar_pairs_empty_list(self, similarity_engine):
        """Test with empty list."""
        result = similarity_engine.find_similar_pairs([])
        assert result == []

    def test_find_similar_pairs_single_note(self, similarity_engine):
        """Test with single note."""
        note = make_note(1, "Test", embedding=np.array([1.0, 0.0, 0.0]))
        result = similarity_engine.find_similar_pairs([note])
        assert result == []

    def test_find_similar_pairs_no_embeddings(self, similarity_engine):
        """Test with notes that have no embeddings."""
        notes = [
            make_note(1, "Note 1", embedding=None),
            make_note(2, "Note 2", embedding=None),
        ]
        result = similarity_engine.find_similar_pairs(notes)
        assert result == []

    def test_find_similar_pairs_similar_notes(self):
        """Test finding similar notes above threshold."""
        engine = SimilarityEngine(threshold=0.9)

        # Create very similar embeddings
        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([0.99, 0.1, 0.0])  # Very similar to emb1
        emb2 = emb2 / np.linalg.norm(emb2)  # Normalize

        notes = [
            make_note(1, "Note 1", category="Tech", embedding=emb1),
            make_note(2, "Note 2", category="Tech", embedding=emb2),
        ]

        result = engine.find_similar_pairs(notes, same_category_only=True)

        assert len(result) == 1
        assert result[0].similarity > 0.9

    def test_find_similar_pairs_different_categories(self):
        """Test that notes with different categories are not matched."""
        engine = SimilarityEngine(threshold=0.5)

        # Create identical embeddings but different categories
        emb = np.array([1.0, 0.0, 0.0])

        notes = [
            make_note(1, "Note 1", category="Tech", embedding=emb),
            make_note(2, "Note 2", category="Science", embedding=emb.copy()),
        ]

        result = engine.find_similar_pairs(notes, same_category_only=True)

        # Should not find pairs due to different categories
        assert len(result) == 0

    def test_find_similar_pairs_same_category_only_false(self):
        """Test with same_category_only=False."""
        engine = SimilarityEngine(threshold=0.5)

        # Create identical embeddings but different categories
        emb = np.array([1.0, 0.0, 0.0])

        notes = [
            make_note(1, "Note 1", category="Tech", embedding=emb),
            make_note(2, "Note 2", category="Science", embedding=emb.copy()),
        ]

        result = engine.find_similar_pairs(notes, same_category_only=False)

        # Should find pairs when same_category_only is False
        assert len(result) == 1

    def test_find_similar_pairs_below_threshold(self):
        """Test that pairs below threshold are not returned."""
        engine = SimilarityEngine(threshold=0.99)

        # Create somewhat similar but not identical embeddings
        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([0.8, 0.6, 0.0])

        notes = [
            make_note(1, "Note 1", category="Test", embedding=emb1),
            make_note(2, "Note 2", category="Test", embedding=emb2),
        ]

        result = engine.find_similar_pairs(notes, same_category_only=True)

        # Similarity is about 0.8, below threshold
        assert len(result) == 0

    def test_find_similar_pairs_sorted_by_similarity(self):
        """Test that pairs are sorted by similarity descending."""
        engine = SimilarityEngine(threshold=0.5)

        emb1 = np.array([1.0, 0.0, 0.0])
        emb2 = np.array([0.9, 0.44, 0.0])  # ~0.9 similarity with emb1
        emb3 = np.array([0.7, 0.71, 0.0])  # ~0.7 similarity with emb1

        notes = [
            make_note(1, "Note 1", category="Test", embedding=emb1),
            make_note(2, "Note 2", category="Test", embedding=emb2),
            make_note(3, "Note 3", category="Test", embedding=emb3),
        ]

        result = engine.find_similar_pairs(notes, same_category_only=True)

        # Should be sorted highest first
        assert len(result) >= 2
        for i in range(len(result) - 1):
            assert result[i].similarity >= result[i + 1].similarity

    def test_find_similar_pairs_exclude_pairs(self):
        """Test excluding specific pairs."""
        engine = SimilarityEngine(threshold=0.5)

        emb = np.array([1.0, 0.0, 0.0])

        notes = [
            make_note(1, "Note 1", category="Test", embedding=emb),
            make_note(2, "Note 2", category="Test", embedding=emb.copy()),
        ]

        # Exclude the only pair
        result = engine.find_similar_pairs(
            notes,
            exclude_pairs={(1, 2)},
            same_category_only=True,
        )

        assert len(result) == 0

    def test_find_similar_pairs_notes_without_category(self):
        """Test that notes without category are skipped when same_category_only=True."""
        engine = SimilarityEngine(threshold=0.5)

        emb = np.array([1.0, 0.0, 0.0])

        notes = [
            make_note(1, "Note 1", category="", embedding=emb),
            make_note(2, "Note 2", category="Test", embedding=emb.copy()),
        ]

        result = engine.find_similar_pairs(notes, same_category_only=True)

        # Note 1 has no category, so no pairs should be found
        assert len(result) == 0


class TestFindSimilarToNote:
    """Tests for finding notes similar to a target."""

    def test_find_similar_to_note_no_embedding(self, similarity_engine):
        """Test with target note having no embedding."""
        target = make_note(1, "Target", embedding=None)
        candidates = [make_note(2, "Candidate", embedding=np.array([1.0, 0.0, 0.0]))]

        result = similarity_engine.find_similar_to_note(target, candidates)
        assert result == []

    def test_find_similar_to_note_excludes_self(self, similarity_engine):
        """Test that target note is excluded from results."""
        emb = np.array([1.0, 0.0, 0.0])
        target = make_note(1, "Target", embedding=emb)
        target.notion_page_id = "same-id"

        candidate = make_note(2, "Candidate", embedding=emb.copy())
        candidate.notion_page_id = "same-id"  # Same as target

        result = similarity_engine.find_similar_to_note(target, [candidate])
        assert len(result) == 0


class TestSimilarityPair:
    """Tests for SimilarityPair dataclass."""

    def test_str_representation(self):
        """Test string representation of pair."""
        note_a = make_note(1, "First Note", embedding=np.array([1.0]))
        note_b = make_note(2, "Second Note", embedding=np.array([1.0]))

        pair = SimilarityPair(note_a=note_a, note_b=note_b, similarity=0.85)

        assert "First Note" in str(pair)
        assert "Second Note" in str(pair)
        assert "0.85" in str(pair)


class TestGroupSimilarNotes:
    """Tests for grouping similar notes."""

    def test_group_similar_notes_empty(self, similarity_engine):
        """Test with empty pairs list."""
        result = similarity_engine.group_similar_notes([])
        assert result == []

    def test_group_similar_notes_single_pair(self, similarity_engine):
        """Test with single pair."""
        note_a = make_note(1, "Note A", embedding=np.array([1.0]))
        note_b = make_note(2, "Note B", embedding=np.array([1.0]))

        pairs = [SimilarityPair(note_a=note_a, note_b=note_b, similarity=0.9)]

        result = similarity_engine.group_similar_notes(pairs)

        assert len(result) == 1
        assert len(result[0]) == 2

    def test_group_similar_notes_transitive(self, similarity_engine):
        """Test that transitive relationships are grouped."""
        note_a = make_note(1, "Note A", embedding=np.array([1.0]))
        note_b = make_note(2, "Note B", embedding=np.array([1.0]))
        note_c = make_note(3, "Note C", embedding=np.array([1.0]))

        # A-B and B-C should result in one group of A, B, C
        pairs = [
            SimilarityPair(note_a=note_a, note_b=note_b, similarity=0.9),
            SimilarityPair(note_a=note_b, note_b=note_c, similarity=0.85),
        ]

        result = similarity_engine.group_similar_notes(pairs)

        assert len(result) == 1
        assert len(result[0]) == 3

    def test_group_similar_notes_separate_groups(self, similarity_engine):
        """Test that unrelated pairs form separate groups."""
        note_a = make_note(1, "Note A", embedding=np.array([1.0]))
        note_b = make_note(2, "Note B", embedding=np.array([1.0]))
        note_c = make_note(3, "Note C", embedding=np.array([1.0]))
        note_d = make_note(4, "Note D", embedding=np.array([1.0]))

        # A-B and C-D are separate groups
        pairs = [
            SimilarityPair(note_a=note_a, note_b=note_b, similarity=0.9),
            SimilarityPair(note_a=note_c, note_b=note_d, similarity=0.85),
        ]

        result = similarity_engine.group_similar_notes(pairs)

        assert len(result) == 2
        assert all(len(group) == 2 for group in result)
