"""Unit tests for EmbeddingService."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from recaller.models.note import Note
from recaller.services.embedding_service import EmbeddingService


@pytest.fixture
def mock_sentence_transformer():
    """Create a mocked SentenceTransformer."""
    with patch(
        "recaller.services.embedding_service.SentenceTransformer"
    ) as mock_class:
        mock_model = MagicMock()
        # Return 384-dim embeddings
        mock_model.encode.return_value = np.random.rand(384).astype(np.float32)
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_class.return_value = mock_model
        yield mock_model


@pytest.fixture
def embedding_service(mock_sentence_transformer):  # noqa: ARG001
    """Create EmbeddingService with mocked model."""
    return EmbeddingService(model_name="test-model")


class TestEmbeddingServiceInit:
    """Tests for EmbeddingService initialization."""

    def test_init_sets_model_name(self):
        """Test that initialization sets model name."""
        with patch("recaller.services.embedding_service.SentenceTransformer"):
            service = EmbeddingService(model_name="custom-model")
            assert service.model_name == "custom-model"

    def test_default_model_name(self):
        """Test default model name."""
        with patch("recaller.services.embedding_service.SentenceTransformer"):
            service = EmbeddingService()
            assert service.model_name == "all-MiniLM-L6-v2"

    def test_lazy_model_loading(self):
        """Test that model is not loaded until first use."""
        with patch(
            "recaller.services.embedding_service.SentenceTransformer"
        ) as mock_class:
            service = EmbeddingService()
            # Model should not be loaded yet
            mock_class.assert_not_called()
            # Access model property to trigger loading
            _ = service.model
            mock_class.assert_called_once_with("all-MiniLM-L6-v2")


class TestGenerateEmbedding:
    """Tests for single embedding generation."""

    def test_generate_embedding_returns_array(
        self, embedding_service, mock_sentence_transformer
    ):
        """Test that generate_embedding returns a numpy array."""
        mock_sentence_transformer.encode.return_value = np.ones(384, dtype=np.float32)

        result = embedding_service.generate_embedding("test text")

        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_generate_embedding_correct_dimension(
        self, embedding_service, mock_sentence_transformer
    ):
        """Test that embedding has correct dimension."""
        mock_sentence_transformer.encode.return_value = np.ones(384, dtype=np.float32)

        result = embedding_service.generate_embedding("test text")

        assert result.shape == (384,)

    def test_generate_embedding_calls_encode(
        self, embedding_service, mock_sentence_transformer
    ):
        """Test that encode is called with correct arguments."""
        embedding_service.generate_embedding("hello world")

        mock_sentence_transformer.encode.assert_called_once_with(
            "hello world", convert_to_numpy=True
        )


class TestGenerateEmbeddings:
    """Tests for batch embedding generation."""

    def test_generate_embeddings_empty_list(self, embedding_service):
        """Test with empty list."""
        result = embedding_service.generate_embeddings([])
        assert result == []

    def test_generate_embeddings_returns_list(
        self, embedding_service, mock_sentence_transformer
    ):
        """Test that batch generation returns a list of arrays."""
        # Mock batch encode to return multiple embeddings
        mock_sentence_transformer.encode.return_value = np.random.rand(3, 384).astype(
            np.float32
        )

        result = embedding_service.generate_embeddings(["text1", "text2", "text3"])

        assert isinstance(result, list)
        assert len(result) == 3
        for emb in result:
            assert isinstance(emb, np.ndarray)
            assert emb.dtype == np.float32


class TestGenerateNoteEmbedding:
    """Tests for note embedding generation."""

    def test_generate_note_embedding_uses_title(
        self, embedding_service, mock_sentence_transformer
    ):
        """Test that note embedding is generated from title."""
        mock_sentence_transformer.encode.return_value = np.ones(384, dtype=np.float32)

        note = Note(
            notion_page_id="test-id",
            title="Test Note Title",
            category="Test",
            source="",
            content="Some content here",
        )

        embedding_service.generate_note_embedding(note)

        mock_sentence_transformer.encode.assert_called_with(
            "Test Note Title", convert_to_numpy=True
        )


class TestGenerateNoteEmbeddings:
    """Tests for batch note embedding generation."""

    def test_generate_note_embeddings_empty_list(self, embedding_service):
        """Test with empty list."""
        result = embedding_service.generate_note_embeddings([])
        assert result == {}

    def test_generate_note_embeddings_skips_notes_without_id(
        self, embedding_service, mock_sentence_transformer
    ):
        """Test that notes without id are skipped."""
        mock_sentence_transformer.encode.return_value = np.random.rand(1, 384).astype(
            np.float32
        )

        note_with_id = Note(
            id=1,
            notion_page_id="id-1",
            title="Note 1",
            category="",
            source="",
            content="",
        )
        note_without_id = Note(
            notion_page_id="id-2",
            title="Note 2",
            category="",
            source="",
            content="",
        )

        result = embedding_service.generate_note_embeddings(
            [note_with_id, note_without_id]
        )

        # Only the note with id should be in the result
        assert 1 in result
        assert len(result) == 1

    def test_generate_note_embeddings_returns_dict(
        self, embedding_service, mock_sentence_transformer
    ):
        """Test that result is a dict mapping id to embedding."""
        mock_sentence_transformer.encode.return_value = np.random.rand(2, 384).astype(
            np.float32
        )

        notes = [
            Note(
                id=1,
                notion_page_id="id-1",
                title="Note 1",
                category="",
                source="",
                content="",
            ),
            Note(
                id=2,
                notion_page_id="id-2",
                title="Note 2",
                category="",
                source="",
                content="",
            ),
        ]

        result = embedding_service.generate_note_embeddings(notes)

        assert isinstance(result, dict)
        assert 1 in result
        assert 2 in result
        assert isinstance(result[1], np.ndarray)


class TestEmbeddingDim:
    """Tests for embedding dimension property."""

    def test_embedding_dim(self, embedding_service, mock_sentence_transformer):
        """Test embedding_dim returns correct value."""
        mock_sentence_transformer.get_sentence_embedding_dimension.return_value = 384

        assert embedding_service.embedding_dim == 384
