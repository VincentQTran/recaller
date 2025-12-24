"""Unit tests for embedding repository operations."""

import numpy as np
import pytest

from recaller.database.repository import Repository


class TestUpsertEmbedding:
    """Tests for upserting embeddings."""

    def test_upsert_creates_new_record(self, repository: Repository):
        """Test creating a new embedding record."""
        embedding = np.random.rand(384).astype(np.float32)

        result = repository.upsert_embedding(
            notion_page_id="test-id-123",
            title="Test Note",
            embedding=embedding,
        )

        assert result is True
        stored = repository.get_embedding_by_notion_id("test-id-123")
        assert stored is not None
        title, stored_embedding = stored
        assert title == "Test Note"
        assert np.allclose(embedding, stored_embedding)

    def test_upsert_updates_on_title_change(self, repository: Repository):
        """Test that embedding is updated when title changes."""
        embedding1 = np.ones(384, dtype=np.float32)
        embedding2 = np.zeros(384, dtype=np.float32)

        repository.upsert_embedding("test-id", "Original Title", embedding1)
        result = repository.upsert_embedding("test-id", "Updated Title", embedding2)

        assert result is True
        title, stored = repository.get_embedding_by_notion_id("test-id")
        assert title == "Updated Title"
        assert np.allclose(stored, embedding2)

    def test_upsert_skips_unchanged_title(self, repository: Repository):
        """Test that unchanged titles don't trigger updates."""
        embedding = np.random.rand(384).astype(np.float32)

        result1 = repository.upsert_embedding("test-id", "Same Title", embedding)
        result2 = repository.upsert_embedding("test-id", "Same Title", embedding)

        assert result1 is True  # First insert
        assert result2 is False  # Skipped (unchanged)

    def test_upsert_preserves_embedding_data(self, repository: Repository):
        """Test that embedding data is preserved correctly."""
        # Create a specific embedding pattern
        embedding = np.arange(384, dtype=np.float32) / 384.0

        repository.upsert_embedding("test-id", "Test", embedding)
        _, stored = repository.get_embedding_by_notion_id("test-id")

        assert stored.shape == (384,)
        assert stored.dtype == np.float32
        assert np.allclose(embedding, stored)


class TestGetEmbeddingByNotionId:
    """Tests for retrieving embeddings by Notion ID."""

    def test_get_existing_embedding(self, repository: Repository):
        """Test retrieving an existing embedding."""
        embedding = np.ones(384, dtype=np.float32) * 0.5

        repository.upsert_embedding("existing-id", "Existing Note", embedding)
        result = repository.get_embedding_by_notion_id("existing-id")

        assert result is not None
        title, stored_embedding = result
        assert title == "Existing Note"
        assert np.allclose(stored_embedding, embedding)

    def test_get_nonexistent_embedding(self, repository: Repository):
        """Test retrieving a non-existent embedding returns None."""
        result = repository.get_embedding_by_notion_id("nonexistent-id")
        assert result is None


class TestGetAllEmbeddings:
    """Tests for retrieving all embeddings."""

    def test_get_all_empty(self, repository: Repository):
        """Test getting all embeddings from empty database."""
        result = repository.get_all_embeddings()
        assert result == []

    def test_get_all_multiple(self, repository: Repository):
        """Test getting all embeddings with multiple records."""
        for i in range(3):
            repository.upsert_embedding(
                f"id-{i}",
                f"Note {i}",
                np.random.rand(384).astype(np.float32),
            )

        result = repository.get_all_embeddings()

        assert len(result) == 3
        notion_ids = {r[0] for r in result}
        assert notion_ids == {"id-0", "id-1", "id-2"}


class TestGetEmbeddingCount:
    """Tests for embedding count."""

    def test_count_empty(self, repository: Repository):
        """Test count with empty database."""
        assert repository.get_embedding_count() == 0

    def test_count_after_inserts(self, repository: Repository):
        """Test count after inserting records."""
        for i in range(5):
            repository.upsert_embedding(
                f"id-{i}",
                f"Note {i}",
                np.zeros(384, dtype=np.float32),
            )

        assert repository.get_embedding_count() == 5


class TestGetEmbeddingsByNotionIds:
    """Tests for bulk fetching embeddings."""

    def test_bulk_fetch_empty_list(self, repository: Repository):
        """Test bulk fetch with empty list."""
        result = repository.get_embeddings_by_notion_ids([])
        assert result == {}

    def test_bulk_fetch_existing_ids(self, repository: Repository):
        """Test bulk fetch with existing IDs."""
        embeddings = {}
        for i in range(3):
            emb = np.random.rand(384).astype(np.float32)
            repository.upsert_embedding(f"id-{i}", f"Note {i}", emb)
            embeddings[f"id-{i}"] = emb

        result = repository.get_embeddings_by_notion_ids(["id-0", "id-2"])

        assert len(result) == 2
        assert "id-0" in result
        assert "id-2" in result
        assert "id-1" not in result

    def test_bulk_fetch_partial_match(self, repository: Repository):
        """Test bulk fetch with some non-existent IDs."""
        repository.upsert_embedding("id-1", "Note 1", np.zeros(384, dtype=np.float32))

        result = repository.get_embeddings_by_notion_ids(
            ["id-1", "nonexistent-1", "nonexistent-2"]
        )

        assert len(result) == 1
        assert "id-1" in result


class TestComputeTitleHash:
    """Tests for title hash computation."""

    def test_same_title_same_hash(self, repository: Repository):
        """Test that same titles produce same hash."""
        hash1 = repository._compute_title_hash("Test Title")
        hash2 = repository._compute_title_hash("Test Title")
        assert hash1 == hash2

    def test_different_title_different_hash(self, repository: Repository):
        """Test that different titles produce different hashes."""
        hash1 = repository._compute_title_hash("Title A")
        hash2 = repository._compute_title_hash("Title B")
        assert hash1 != hash2

    def test_hash_length(self, repository: Repository):
        """Test that hash has correct length (SHA256 = 64 hex chars)."""
        hash_value = repository._compute_title_hash("Any Title")
        assert len(hash_value) == 64
