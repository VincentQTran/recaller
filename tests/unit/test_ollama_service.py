"""Unit tests for OllamaService."""

from unittest.mock import MagicMock, patch

import pytest

from recaller.services.ollama_service import OllamaError, OllamaService


@pytest.fixture
def mock_client():
    """Create a mocked Ollama Client."""
    with patch("recaller.services.ollama_service.Client") as mock:
        yield mock


@pytest.fixture
def ollama_service(mock_client):
    """Create an OllamaService instance with mocked client."""
    mock_instance = MagicMock()
    mock_client.return_value = mock_instance
    service = OllamaService(
        model_name="gpt-oss:120b-cloud",
        host="https://ollama.com",
        api_key="test-api-key",
    )
    return service


class TestOllamaServiceInit:
    """Tests for OllamaService initialization."""

    def test_init_default_values(self, mock_client):
        """Test default initialization."""
        mock_client.return_value = MagicMock()
        service = OllamaService()
        assert service.model_name == "gpt-oss:120b-cloud"
        assert service.host == "https://ollama.com"

    def test_init_custom_values(self, mock_client):
        """Test custom initialization."""
        mock_client.return_value = MagicMock()
        service = OllamaService(
            model_name="mistral",
            host="https://custom.ollama.com",
            api_key="custom-key",
        )
        assert service.model_name == "mistral"
        assert service.host == "https://custom.ollama.com"
        assert service.api_key == "custom-key"

    def test_init_creates_client_with_auth(self, mock_client):
        """Test that client is created with authorization header."""
        mock_client.return_value = MagicMock()
        OllamaService(api_key="test-key")
        mock_client.assert_called_once()
        call_kwargs = mock_client.call_args[1]
        assert call_kwargs["headers"]["Authorization"] == "Bearer test-key"


class TestCheckConnection:
    """Tests for connection checking."""

    def test_check_connection_success(self, ollama_service):
        """Test successful connection check."""
        ollama_service._client.list.return_value = {"models": []}
        assert ollama_service.check_connection() is True

    def test_check_connection_failure(self, ollama_service):
        """Test failed connection check."""
        ollama_service._client.list.side_effect = Exception("Connection error")
        assert ollama_service.check_connection() is False


class TestListModels:
    """Tests for listing models."""

    def test_list_models_success(self, ollama_service):
        """Test successful model listing."""
        ollama_service._client.list.return_value = {
            "models": [
                {"name": "gpt-oss:120b-cloud"},
                {"name": "mistral"},
                {"name": "codellama"},
            ]
        }

        models = ollama_service.list_models()

        assert models == ["gpt-oss:120b-cloud", "mistral", "codellama"]

    def test_list_models_empty(self, ollama_service):
        """Test empty model list."""
        ollama_service._client.list.return_value = {"models": []}

        models = ollama_service.list_models()

        assert models == []

    def test_list_models_error(self, ollama_service):
        """Test model listing with error."""
        ollama_service._client.list.side_effect = Exception("Error")

        models = ollama_service.list_models()

        assert models == []


class TestGenerate:
    """Tests for content generation."""

    def test_generate_success(self, ollama_service):
        """Test successful generation."""
        ollama_service._client.chat.return_value = {
            "message": {"content": "Generated text response"}
        }

        result = ollama_service.generate("Test prompt")

        assert result == "Generated text response"
        ollama_service._client.chat.assert_called_once()
        call_kwargs = ollama_service._client.chat.call_args[1]
        assert call_kwargs["model"] == "gpt-oss:120b-cloud"
        assert call_kwargs["stream"] is False
        assert len(call_kwargs["messages"]) == 1
        assert call_kwargs["messages"][0]["role"] == "user"
        assert call_kwargs["messages"][0]["content"] == "Test prompt"

    def test_generate_with_system_prompt(self, ollama_service):
        """Test generation with system prompt."""
        ollama_service._client.chat.return_value = {
            "message": {"content": "Response with system"}
        }

        result = ollama_service.generate("User prompt", system="System prompt")

        assert result == "Response with system"
        call_kwargs = ollama_service._client.chat.call_args[1]
        assert len(call_kwargs["messages"]) == 2
        assert call_kwargs["messages"][0]["role"] == "system"
        assert call_kwargs["messages"][0]["content"] == "System prompt"
        assert call_kwargs["messages"][1]["role"] == "user"

    def test_generate_strips_response(self, ollama_service):
        """Test that response is stripped."""
        ollama_service._client.chat.return_value = {
            "message": {"content": "  Response with whitespace  \n"}
        }

        result = ollama_service.generate("Test")

        assert result == "Response with whitespace"

    def test_generate_connection_error(self, ollama_service):
        """Test generation with connection error."""
        ollama_service._client.chat.side_effect = Exception("connection refused")

        with pytest.raises(OllamaError, match="Cannot connect"):
            ollama_service.generate("Test")

    def test_generate_timeout_error(self, ollama_service):
        """Test generation with timeout."""
        ollama_service._client.chat.side_effect = Exception("timeout exceeded")

        with pytest.raises(OllamaError, match="timed out"):
            ollama_service.generate("Test")

    def test_generate_other_error(self, ollama_service):
        """Test generation with general error."""
        ollama_service._client.chat.side_effect = Exception("Some other error")

        with pytest.raises(OllamaError, match="request failed"):
            ollama_service.generate("Test")


class TestGenerateCombinedTitle:
    """Tests for combined title generation."""

    def test_generate_combined_title(self, ollama_service):
        """Test generating combined title."""
        ollama_service._client.chat.return_value = {
            "message": {"content": "Combined Title"}
        }

        result = ollama_service.generate_combined_title("Title A", "Title B")

        assert result == "Combined Title"

    def test_generate_combined_title_strips_quotes(self, ollama_service):
        """Test that quotes are stripped from title."""
        ollama_service._client.chat.return_value = {
            "message": {"content": '"Quoted Title"'}
        }

        result = ollama_service.generate_combined_title("A", "B")

        assert result == "Quoted Title"

    def test_generate_combined_title_takes_first_line(self, ollama_service):
        """Test that only first line is returned."""
        ollama_service._client.chat.return_value = {
            "message": {"content": "First Line\nSecond Line\nThird"}
        }

        result = ollama_service.generate_combined_title("A", "B")

        assert result == "First Line"


class TestGenerateCombinedTitleForGroup:
    """Tests for group title generation."""

    def test_single_title(self, ollama_service):
        """Test with single title returns it directly."""
        result = ollama_service.generate_combined_title_for_group(["Single Title"])
        assert result == "Single Title"

    def test_two_titles(self, ollama_service):
        """Test with two titles uses generate_combined_title."""
        ollama_service._client.chat.return_value = {
            "message": {"content": "Combined Two"}
        }

        result = ollama_service.generate_combined_title_for_group(["Title A", "Title B"])

        assert result == "Combined Two"

    def test_multiple_titles(self, ollama_service):
        """Test with multiple titles."""
        ollama_service._client.chat.return_value = {
            "message": {"content": "Combined Multiple"}
        }

        result = ollama_service.generate_combined_title_for_group(
            ["Title A", "Title B", "Title C"]
        )

        assert result == "Combined Multiple"


class TestGenerateFlashcardsJson:
    """Tests for flashcard JSON generation."""

    def test_generate_flashcards_json(self, ollama_service):
        """Test generating flashcards JSON."""
        expected_json = '[{"front": "Q?", "back": "A", "type": "basic"}]'
        ollama_service._client.chat.return_value = {
            "message": {"content": expected_json}
        }

        result = ollama_service.generate_flashcards_json(
            title="Test Note",
            category="Science",
            source="Textbook",
            content="This is the content.",
            min_cards=1,
            max_cards=3,
        )

        assert result == expected_json

    def test_generate_flashcards_json_truncates_content(self, ollama_service):
        """Test that long content is truncated."""
        long_content = "x" * 3000
        ollama_service._client.chat.return_value = {
            "message": {"content": "[]"}
        }

        ollama_service.generate_flashcards_json(
            title="Test",
            category="Cat",
            source="Src",
            content=long_content,
        )

        call_kwargs = ollama_service._client.chat.call_args[1]
        prompt = call_kwargs["messages"][-1]["content"]
        # Content should be truncated to 2000 chars in prompt
        assert len(prompt) < len(long_content)


class TestOllamaError:
    """Tests for OllamaError exception."""

    def test_ollama_error_message(self):
        """Test error message."""
        error = OllamaError("Test error message")
        assert str(error) == "Test error message"
