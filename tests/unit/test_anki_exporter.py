"""Unit tests for AnkiExporter."""

from unittest.mock import MagicMock, patch

import pytest

from recaller.models.flashcard import ExportStatus, Flashcard, FlashcardType
from recaller.services.anki_exporter import AnkiConnectError, AnkiExporter, ExportResult


def make_flashcard(
    front: str = "Question",
    back: str = "Answer",
    card_type: FlashcardType = FlashcardType.BASIC,
    deck_name: str = "Test::Deck",
    tags: list = None,
) -> Flashcard:
    """Helper to create a test flashcard."""
    return Flashcard(
        front=front,
        back=back,
        card_type=card_type,
        deck_name=deck_name,
        tags=tags or ["test"],
        note_id=1,
    )


@pytest.fixture
def mock_requests():
    """Mock the requests module."""
    with patch("recaller.services.anki_exporter.requests") as mock:
        yield mock


@pytest.fixture
def anki_exporter():
    """Create an AnkiExporter instance."""
    return AnkiExporter(url="http://localhost:8765")


class TestAnkiExporterInit:
    """Tests for AnkiExporter initialization."""

    def test_init_default_url(self):
        """Test default URL."""
        exporter = AnkiExporter()
        assert exporter.url == "http://localhost:8765"

    def test_init_custom_url(self):
        """Test custom URL."""
        exporter = AnkiExporter(url="http://custom:9999")
        assert exporter.url == "http://custom:9999"


class TestInvoke:
    """Tests for the _invoke method."""

    def test_invoke_success(self, anki_exporter, mock_requests):
        """Test successful invocation."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": "test_result", "error": None}
        mock_requests.post.return_value = mock_response

        result = anki_exporter._invoke("testAction", param1="value1")

        assert result == "test_result"
        mock_requests.post.assert_called_once()

    def test_invoke_with_error(self, anki_exporter, mock_requests):
        """Test invocation with API error."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": None, "error": "API Error"}
        mock_requests.post.return_value = mock_response

        with pytest.raises(AnkiConnectError, match="API Error"):
            anki_exporter._invoke("testAction")

    def test_invoke_connection_error(self, anki_exporter, mock_requests):
        """Test invocation with connection error."""
        import requests

        mock_requests.post.side_effect = requests.exceptions.ConnectionError()
        mock_requests.exceptions = requests.exceptions

        with pytest.raises(AnkiConnectError, match="Cannot connect"):
            anki_exporter._invoke("testAction")

    def test_invoke_timeout(self, anki_exporter, mock_requests):
        """Test invocation with timeout."""
        import requests

        mock_requests.post.side_effect = requests.exceptions.Timeout()
        mock_requests.exceptions = requests.exceptions

        with pytest.raises(AnkiConnectError, match="timed out"):
            anki_exporter._invoke("testAction")


class TestCheckConnection:
    """Tests for connection checking."""

    def test_check_connection_success(self, anki_exporter, mock_requests):
        """Test successful connection check."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": 6, "error": None}
        mock_requests.post.return_value = mock_response

        assert anki_exporter.check_connection() is True

    def test_check_connection_failure(self, anki_exporter, mock_requests):
        """Test failed connection check."""
        import requests

        mock_requests.post.side_effect = requests.exceptions.ConnectionError()
        mock_requests.exceptions = requests.exceptions

        assert anki_exporter.check_connection() is False


class TestGetDeckNames:
    """Tests for getting deck names."""

    def test_get_deck_names(self, anki_exporter, mock_requests):
        """Test getting deck names."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": ["Default", "Test::Deck"],
            "error": None,
        }
        mock_requests.post.return_value = mock_response

        result = anki_exporter.get_deck_names()

        assert result == ["Default", "Test::Deck"]


class TestEnsureDeckExists:
    """Tests for ensuring deck exists."""

    def test_ensure_deck_exists_already_exists(self, anki_exporter, mock_requests):
        """Test when deck already exists."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "result": ["Default", "Test::Deck"],
            "error": None,
        }
        mock_requests.post.return_value = mock_response

        result = anki_exporter.ensure_deck_exists("Test::Deck")

        assert result is True

    def test_ensure_deck_exists_creates_new(self, anki_exporter, mock_requests):
        """Test creating new deck."""
        # First call returns existing decks, second call creates new deck
        mock_requests.post.return_value.json.side_effect = [
            {"result": ["Default"], "error": None},  # deckNames
            {"result": 123, "error": None},  # createDeck
        ]

        result = anki_exporter.ensure_deck_exists("New::Deck")

        assert result is True
        assert mock_requests.post.call_count == 2


class TestFlashcardToAnkiNote:
    """Tests for flashcard conversion."""

    def test_basic_flashcard_conversion(self, anki_exporter):
        """Test converting basic flashcard."""
        flashcard = make_flashcard(front="Q", back="A")

        result = anki_exporter._flashcard_to_anki_note(flashcard)

        assert result["modelName"] == "Basic"
        assert result["fields"]["Front"] == "Q"
        assert result["fields"]["Back"] == "A"
        assert result["deckName"] == "Test::Deck"
        assert result["tags"] == ["test"]

    def test_cloze_flashcard_conversion(self, anki_exporter):
        """Test converting cloze flashcard."""
        flashcard = make_flashcard(
            front="The {{c1::answer}} is here",
            back="answer",
            card_type=FlashcardType.CLOZE,
        )

        result = anki_exporter._flashcard_to_anki_note(flashcard)

        assert result["modelName"] == "Cloze"
        assert result["fields"]["Text"] == "The {{c1::answer}} is here"
        assert result["fields"]["Extra"] == "answer"


class TestAddNote:
    """Tests for adding single note."""

    def test_add_note_success(self, anki_exporter, mock_requests):
        """Test successful note addition."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": 12345, "error": None}
        mock_requests.post.return_value = mock_response

        flashcard = make_flashcard()
        result = anki_exporter.add_note(flashcard)

        assert result.success is True
        assert result.anki_note_id == 12345
        assert flashcard.anki_note_id == 12345
        assert flashcard.export_status == ExportStatus.EXPORTED

    def test_add_note_failure(self, anki_exporter, mock_requests):
        """Test failed note addition."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": None, "error": "Duplicate note"}
        mock_requests.post.return_value = mock_response

        flashcard = make_flashcard()
        result = anki_exporter.add_note(flashcard)

        assert result.success is False
        assert "Duplicate" in result.error
        assert flashcard.export_status == ExportStatus.FAILED


class TestAddNotes:
    """Tests for adding multiple notes."""

    def test_add_notes_success(self, anki_exporter, mock_requests):
        """Test successful batch addition."""
        # Mock responses for deckNames and addNotes
        mock_requests.post.return_value.json.side_effect = [
            {"result": ["Test::Deck"], "error": None},  # deckNames
            {"result": [111, 222, 333], "error": None},  # addNotes
        ]

        flashcards = [make_flashcard() for _ in range(3)]
        results = anki_exporter.add_notes(flashcards)

        assert len(results) == 3
        assert all(r.success for r in results)
        assert results[0].anki_note_id == 111
        assert results[1].anki_note_id == 222
        assert results[2].anki_note_id == 333

    def test_add_notes_partial_failure(self, anki_exporter, mock_requests):
        """Test batch with some failures."""
        mock_requests.post.return_value.json.side_effect = [
            {"result": ["Test::Deck"], "error": None},  # deckNames
            {"result": [111, None, 333], "error": None},  # addNotes (one failed)
        ]

        flashcards = [make_flashcard() for _ in range(3)]
        results = anki_exporter.add_notes(flashcards)

        assert results[0].success is True
        assert results[1].success is False
        assert results[2].success is True

    def test_add_notes_empty_list(self, anki_exporter):
        """Test with empty flashcard list."""
        results = anki_exporter.add_notes([])
        assert results == []

    def test_add_notes_creates_deck(self, anki_exporter, mock_requests):
        """Test that missing deck is created."""
        mock_requests.post.return_value.json.side_effect = [
            {"result": ["Default"], "error": None},  # deckNames - deck doesn't exist
            {"result": 123, "error": None},  # createDeck
            {"result": [111], "error": None},  # addNotes
        ]

        flashcards = [make_flashcard(deck_name="New::Deck")]
        results = anki_exporter.add_notes(flashcards)

        assert len(results) == 1
        assert results[0].success is True


class TestExportResult:
    """Tests for ExportResult dataclass."""

    def test_export_result_success(self):
        """Test successful export result."""
        flashcard = make_flashcard()
        result = ExportResult(
            flashcard=flashcard, success=True, anki_note_id=123
        )

        assert result.success is True
        assert result.anki_note_id == 123
        assert result.error is None

    def test_export_result_failure(self):
        """Test failed export result."""
        flashcard = make_flashcard()
        result = ExportResult(
            flashcard=flashcard, success=False, error="Duplicate"
        )

        assert result.success is False
        assert result.anki_note_id is None
        assert result.error == "Duplicate"


class TestSync:
    """Tests for Anki sync."""

    def test_sync_success(self, anki_exporter, mock_requests):
        """Test successful sync."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": None, "error": None}
        mock_requests.post.return_value = mock_response

        result = anki_exporter.sync()

        assert result is True

    def test_sync_failure(self, anki_exporter, mock_requests):
        """Test failed sync."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"result": None, "error": "Sync failed"}
        mock_requests.post.return_value = mock_response

        result = anki_exporter.sync()

        assert result is False
