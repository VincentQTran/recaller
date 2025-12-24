"""Unit tests for FlashcardGenerator."""

import json
from unittest.mock import MagicMock

import pytest

from recaller.models.flashcard import Flashcard, FlashcardType
from recaller.models.note import Note
from recaller.services.flashcard_generator import FlashcardGenerator


def make_note(
    note_id: int = 1,
    title: str = "Test Note",
    category: str = "Technology",
    source: str = "Wikipedia",
    content: str = "Test content for the note.",
) -> Note:
    """Helper to create a test note."""
    return Note(
        id=note_id,
        notion_page_id=f"page-{note_id}",
        title=title,
        category=category,
        source=source,
        content=content,
    )


@pytest.fixture
def mock_ollama():
    """Create a mocked OllamaService."""
    mock = MagicMock()
    mock.model_name = "gpt-oss:120b-cloud"
    return mock


@pytest.fixture
def flashcard_generator(mock_ollama):
    """Create FlashcardGenerator with mocked Ollama service."""
    return FlashcardGenerator(
        ollama_service=mock_ollama,
        cards_per_note_min=1,
        cards_per_note_max=3,
    )


class TestFlashcardGeneratorInit:
    """Tests for FlashcardGenerator initialization."""

    def test_init_stores_settings(self, mock_ollama):
        """Test that initialization stores settings correctly."""
        gen = FlashcardGenerator(
            ollama_service=mock_ollama,
            cards_per_note_min=2,
            cards_per_note_max=5,
        )

        assert gen.cards_min == 2
        assert gen.cards_max == 5

    def test_init_default_values(self, mock_ollama):
        """Test default values for optional parameters."""
        gen = FlashcardGenerator(ollama_service=mock_ollama)

        assert gen.cards_min == 1
        assert gen.cards_max == 3


class TestGenerateFlashcards:
    """Tests for flashcard generation."""

    def test_generate_flashcards_returns_list(self, flashcard_generator, mock_ollama):
        """Test that generate_flashcards returns a list of Flashcard objects."""
        mock_ollama.generate.return_value = json.dumps([
            {"front": "What is Python?", "back": "A programming language", "type": "basic"}
        ])

        note = make_note()
        result = flashcard_generator.generate_flashcards(note)

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], Flashcard)

    def test_generate_flashcards_content(self, flashcard_generator, mock_ollama):
        """Test that flashcard content is extracted correctly."""
        mock_ollama.generate.return_value = json.dumps([
            {"front": "Question 1", "back": "Answer 1", "type": "basic"},
            {"front": "Question 2", "back": "Answer 2", "type": "cloze"},
        ])

        note = make_note()
        result = flashcard_generator.generate_flashcards(note)

        assert result[0].front == "Question 1"
        assert result[0].back == "Answer 1"
        assert result[0].card_type == FlashcardType.BASIC

        assert result[1].front == "Question 2"
        assert result[1].back == "Answer 2"
        assert result[1].card_type == FlashcardType.CLOZE

    def test_generate_flashcards_sets_metadata(self, flashcard_generator, mock_ollama):
        """Test that flashcard metadata is set correctly."""
        mock_ollama.generate.return_value = json.dumps([
            {"front": "Q", "back": "A", "type": "basic"}
        ])

        note = make_note(note_id=42)
        result = flashcard_generator.generate_flashcards(note)

        assert result[0].note_id == 42
        # Deck name is now based on current date (Recaller::MM-DD-YYYY format)
        import re
        assert re.match(r"Recaller::\d{2}-\d{2}-\d{4}", result[0].deck_name)

    def test_generate_flashcards_with_code_block(self, flashcard_generator, mock_ollama):
        """Test parsing JSON from code block."""
        mock_ollama.generate.return_value = """Here are the flashcards:
```json
[{"front": "Q", "back": "A", "type": "basic"}]
```"""

        note = make_note()
        result = flashcard_generator.generate_flashcards(note)

        assert len(result) == 1
        assert result[0].front == "Q"


class TestBuildPrompt:
    """Tests for prompt building."""

    def test_prompt_contains_note_title(self, flashcard_generator):
        """Test that prompt includes note title."""
        note = make_note(title="Machine Learning Basics")
        prompt = flashcard_generator._build_prompt(note)

        assert "Machine Learning Basics" in prompt

    def test_prompt_contains_category(self, flashcard_generator):
        """Test that prompt includes category."""
        note = make_note(category="Data Science")
        prompt = flashcard_generator._build_prompt(note)

        assert "Data Science" in prompt

    def test_prompt_contains_card_counts(self, flashcard_generator):
        """Test that prompt includes card count range."""
        note = make_note()
        prompt = flashcard_generator._build_prompt(note)

        assert "1" in prompt  # cards_min
        assert "3" in prompt  # cards_max

    def test_prompt_truncates_long_content(self, flashcard_generator):
        """Test that long content is truncated."""
        long_content = "x" * 3000
        note = make_note(content=long_content)
        prompt = flashcard_generator._build_prompt(note)

        # Content should be truncated to 2000 chars
        assert len(prompt) < 3000 + 500  # Some overhead for prompt template


class TestParseResponse:
    """Tests for response parsing."""

    def test_parse_valid_json(self, flashcard_generator):
        """Test parsing valid JSON array."""
        response = '[{"front": "Q", "back": "A", "type": "basic"}]'
        result = flashcard_generator._parse_response(response)

        assert len(result) == 1
        assert result[0]["front"] == "Q"

    def test_parse_json_in_code_block(self, flashcard_generator):
        """Test parsing JSON from markdown code block."""
        response = """Some text
```json
[{"front": "Q", "back": "A", "type": "basic"}]
```
More text"""
        result = flashcard_generator._parse_response(response)

        assert len(result) == 1

    def test_parse_invalid_json_raises(self, flashcard_generator):
        """Test that invalid JSON raises ValueError."""
        response = "not json at all"

        with pytest.raises(ValueError, match="Could not extract JSON"):
            flashcard_generator._parse_response(response)

    def test_parse_non_array_raises(self, flashcard_generator):
        """Test that non-array JSON raises ValueError."""
        response = '{"front": "Q", "back": "A"}'

        with pytest.raises(ValueError, match="Could not extract JSON"):
            flashcard_generator._parse_response(response)


class TestCreateFlashcards:
    """Tests for flashcard object creation."""

    def test_create_flashcards_basic(self, flashcard_generator):
        """Test creating basic flashcards."""
        cards_data = [
            {"front": "Q1", "back": "A1", "type": "basic"},
            {"front": "Q2", "back": "A2", "type": "basic"},
        ]
        note = make_note()

        result = flashcard_generator._create_flashcards(cards_data, note)

        assert len(result) == 2
        assert all(c.card_type == FlashcardType.BASIC for c in result)

    def test_create_flashcards_cloze(self, flashcard_generator):
        """Test creating cloze flashcards."""
        cards_data = [{"front": "The {{c1::answer}}", "back": "answer", "type": "cloze"}]
        note = make_note()

        result = flashcard_generator._create_flashcards(cards_data, note)

        assert result[0].card_type == FlashcardType.CLOZE

    def test_create_flashcards_skips_empty(self, flashcard_generator):
        """Test that empty front/back cards are skipped."""
        cards_data = [
            {"front": "", "back": "A1", "type": "basic"},
            {"front": "Q2", "back": "", "type": "basic"},
            {"front": "Q3", "back": "A3", "type": "basic"},
        ]
        note = make_note()

        result = flashcard_generator._create_flashcards(cards_data, note)

        assert len(result) == 1
        assert result[0].front == "Q3"


class TestGenerateTags:
    """Tests for tag generation."""

    def test_tags_include_recaller(self, flashcard_generator):
        """Test that 'recaller' tag is always included."""
        note = make_note()
        tags = flashcard_generator._generate_tags(note)

        assert "recaller" in tags

    def test_tags_include_category(self, flashcard_generator):
        """Test that category is included as tag."""
        note = make_note(category="Machine Learning")
        tags = flashcard_generator._generate_tags(note)

        assert "category::machine_learning" in tags

    def test_tags_include_source(self, flashcard_generator):
        """Test that source is included as tag."""
        note = make_note(source="Wikipedia")
        tags = flashcard_generator._generate_tags(note)

        assert "source::wikipedia" in tags

    def test_tags_skip_long_source(self, flashcard_generator):
        """Test that long sources are not included as tags."""
        # Source must be > 50 chars to be skipped
        long_source = "A" * 60
        note = make_note(source=long_source)
        tags = flashcard_generator._generate_tags(note)

        # Should only have recaller and category tags
        source_tags = [t for t in tags if t.startswith("source::")]
        assert len(source_tags) == 0

    def test_tags_handle_empty_category(self, flashcard_generator):
        """Test handling of empty category."""
        note = make_note(category="")
        tags = flashcard_generator._generate_tags(note)

        category_tags = [t for t in tags if t.startswith("category::")]
        assert len(category_tags) == 0


class TestGenerateFlashcardsBatch:
    """Tests for batch flashcard generation."""

    def test_batch_returns_dict(self, flashcard_generator, mock_ollama):
        """Test that batch generation returns a dictionary."""
        mock_ollama.generate.return_value = json.dumps([
            {"front": "Q", "back": "A", "type": "basic"}
        ])

        notes = [make_note(note_id=1), make_note(note_id=2)]
        result = flashcard_generator.generate_flashcards_batch(notes)

        assert isinstance(result, dict)
        assert 1 in result
        assert 2 in result

    def test_batch_skips_notes_without_id(self, flashcard_generator, mock_ollama):
        """Test that notes without ID are skipped."""
        note_with_id = make_note(note_id=1)
        note_without_id = Note(
            notion_page_id="page-x",
            title="No ID",
            category="",
            source="",
            content="",
        )

        mock_ollama.generate.return_value = json.dumps([
            {"front": "Q", "back": "A", "type": "basic"}
        ])

        result = flashcard_generator.generate_flashcards_batch([note_with_id, note_without_id])

        assert 1 in result
        assert None not in result

    def test_batch_handles_failures(self, flashcard_generator, mock_ollama):
        """Test that batch handles failures gracefully."""
        mock_ollama.generate.side_effect = Exception("API Error")

        notes = [make_note(note_id=1)]
        result = flashcard_generator.generate_flashcards_batch(notes)

        # Should return empty list for failed note
        assert result[1] == []
