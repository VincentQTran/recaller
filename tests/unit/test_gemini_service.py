"""Unit tests for GeminiService."""

from unittest.mock import MagicMock, patch

import pytest

from recaller.services.gemini_service import GeminiService


@pytest.fixture
def mock_genai():
    """Mock the google.genai module."""
    with patch("recaller.services.gemini_service.genai") as mock:
        mock_client = MagicMock()
        mock.Client.return_value = mock_client
        yield mock, mock_client


@pytest.fixture
def gemini_service(mock_genai):
    """Create a GeminiService with mocked Gemini API."""
    return GeminiService(api_key="test-api-key")


class TestGeminiServiceInit:
    """Tests for GeminiService initialization."""

    def test_init_stores_api_key(self, mock_genai):
        """Test that initialization stores the API key."""
        service = GeminiService(api_key="test-key")
        assert service.api_key == "test-key"

    def test_init_default_model(self, mock_genai):
        """Test default model name."""
        service = GeminiService(api_key="test-key")
        assert service.model_name == "gemini-2.0-flash-lite"

    def test_init_custom_model(self, mock_genai):
        """Test custom model name."""
        service = GeminiService(api_key="test-key", model_name="gemini-pro")
        assert service.model_name == "gemini-pro"

    def test_lazy_client_loading(self, mock_genai):
        """Test that client is not loaded until first use."""
        mock, _ = mock_genai
        service = GeminiService(api_key="test-key")

        # Client should not be created yet
        mock.Client.assert_not_called()

        # Access client property to trigger loading
        _ = service.client
        mock.Client.assert_called_once_with(api_key="test-key")


class TestGenerateCombinedTitle:
    """Tests for combined title generation."""

    def test_generate_combined_title_returns_stripped_text(
        self, gemini_service, mock_genai
    ):
        """Test that generated title is stripped of whitespace and quotes."""
        _, mock_client = mock_genai
        mock_response = MagicMock()
        mock_response.text = '  "Combined Topic"  '
        mock_client.models.generate_content.return_value = mock_response

        result = gemini_service.generate_combined_title("Title A", "Title B")

        assert result == "Combined Topic"

    def test_generate_combined_title_calls_model(self, gemini_service, mock_genai):
        """Test that model is called with appropriate prompt."""
        _, mock_client = mock_genai
        mock_response = MagicMock()
        mock_response.text = "Combined Title"
        mock_client.models.generate_content.return_value = mock_response

        gemini_service.generate_combined_title("First Note", "Second Note")

        # Verify model was called
        mock_client.models.generate_content.assert_called_once()
        call_kwargs = mock_client.models.generate_content.call_args
        prompt = call_kwargs.kwargs["contents"]

        # Check prompt contains both titles
        assert "First Note" in prompt
        assert "Second Note" in prompt

    def test_generate_combined_title_uses_correct_model(
        self, gemini_service, mock_genai
    ):
        """Test that the correct model is specified."""
        _, mock_client = mock_genai
        mock_response = MagicMock()
        mock_response.text = "Combined Title"
        mock_client.models.generate_content.return_value = mock_response

        gemini_service.generate_combined_title("Title A", "Title B")

        call_kwargs = mock_client.models.generate_content.call_args
        assert call_kwargs.kwargs["model"] == "gemini-2.0-flash-lite"


class TestGenerateCombinedTitleForGroup:
    """Tests for group title generation."""

    def test_single_title_returns_as_is(self, gemini_service, mock_genai):
        """Test that single title is returned unchanged."""
        result = gemini_service.generate_combined_title_for_group(["Only Title"])
        assert result == "Only Title"

    def test_two_titles_calls_combined_title(self, gemini_service, mock_genai):
        """Test that two titles uses generate_combined_title."""
        _, mock_client = mock_genai
        mock_response = MagicMock()
        mock_response.text = "Combined"
        mock_client.models.generate_content.return_value = mock_response

        result = gemini_service.generate_combined_title_for_group(
            ["Title A", "Title B"]
        )

        assert result == "Combined"
        mock_client.models.generate_content.assert_called_once()

    def test_multiple_titles_generates_group_title(self, gemini_service, mock_genai):
        """Test that multiple titles generates group title."""
        _, mock_client = mock_genai
        mock_response = MagicMock()
        mock_response.text = "Group Title"
        mock_client.models.generate_content.return_value = mock_response

        result = gemini_service.generate_combined_title_for_group(
            ["Title A", "Title B", "Title C"]
        )

        assert result == "Group Title"
        call_kwargs = mock_client.models.generate_content.call_args
        prompt = call_kwargs.kwargs["contents"]

        # Check prompt contains all titles
        assert "Title A" in prompt
        assert "Title B" in prompt
        assert "Title C" in prompt
