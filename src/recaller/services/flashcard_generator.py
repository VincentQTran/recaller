"""Flashcard generator service using Ollama."""

import json
import re
from typing import Any

from tenacity import retry, stop_after_attempt, wait_exponential

from recaller.models.flashcard import Flashcard, FlashcardType
from recaller.models.note import Note
from recaller.services.ollama_service import OllamaService


class FlashcardGenerator:
    """Service for generating flashcards from notes using Ollama.

    Generates 1-3 flashcards per note with structured JSON output.
    """

    def __init__(
        self,
        ollama_service: OllamaService,
        cards_per_note_min: int = 1,
        cards_per_note_max: int = 3,
        deck_name: str = "Recaller::Weekly",
    ):
        """Initialize the flashcard generator.

        Args:
            ollama_service: OllamaService instance for LLM calls
            cards_per_note_min: Minimum flashcards to generate per note
            cards_per_note_max: Maximum flashcards to generate per note
            deck_name: Default Anki deck name for cards
        """
        self.llm = ollama_service
        self.cards_min = cards_per_note_min
        self.cards_max = cards_per_note_max
        self.deck_name = deck_name

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def generate_flashcards(self, note: Note) -> list[Flashcard]:
        """Generate flashcards for a single note.

        Args:
            note: The note to generate flashcards from

        Returns:
            List of Flashcard objects
        """
        prompt = self._build_prompt(note)
        system = (
            "You are an expert educator creating flashcards for spaced repetition. "
            "Always respond with valid JSON only, no markdown or explanation."
        )
        response_text = self.llm.generate(prompt, system=system)
        cards_data = self._parse_response(response_text)

        return self._create_flashcards(cards_data, note)

    def generate_flashcards_batch(self, notes: list[Note]) -> dict[int, list[Flashcard]]:
        """Generate flashcards for multiple notes.

        Args:
            notes: List of notes to process

        Returns:
            Dictionary mapping note_id to list of flashcards
        """
        results: dict[int, list[Flashcard]] = {}

        for note in notes:
            if note.id is None:
                continue

            try:
                flashcards = self.generate_flashcards(note)
                results[note.id] = flashcards
            except Exception:
                # Skip failed notes, can be retried later
                results[note.id] = []

        return results

    def _build_prompt(self, note: Note) -> str:
        """Build the prompt for flashcard generation.

        Args:
            note: The source note

        Returns:
            Formatted prompt string
        """
        content_preview = note.content[:2000] if len(note.content) > 2000 else note.content

        prompt = f"""You are an expert educator creating flashcards for spaced repetition learning.

Generate {self.cards_min}-{self.cards_max} high-quality flashcards from the following note.

**Note Title:** {note.title}
**Category:** {note.category or "General"}
**Source:** {note.source or "Unknown"}

**Content:**
{content_preview}

**Requirements:**
1. Create {self.cards_min} to {self.cards_max} flashcards that test key concepts
2. Each card should test ONE specific concept or fact
3. Questions should be clear and unambiguous
4. Answers should be concise but complete
5. Focus on the most important and memorable information
6. Avoid trivial or obvious questions

**Output Format:**
Return a JSON array of flashcard objects. Each object must have:
- "front": The question or prompt (string)
- "back": The answer (string)
- "type": Either "basic" or "cloze" (string)

Example output:
```json
[
  {{"front": "What is the capital of France?", "back": "Paris", "type": "basic"}},
  {{"front": "The {{{{c1::mitochondria}}}} is the powerhouse of the cell.",
    "back": "mitochondria", "type": "cloze"}}
]
```

Generate the flashcards now:"""

        return prompt

    def _parse_response(self, response_text: str) -> list[dict[str, Any]]:
        """Parse the LLM response to extract flashcard data.

        Args:
            response_text: Raw response from LLM

        Returns:
            List of flashcard dictionaries
        """
        # Try to extract JSON from response
        json_match = re.search(r"\[[\s\S]*\]", response_text)
        if not json_match:
            # Try to find JSON in code blocks
            code_block_match = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", response_text)
            if code_block_match:
                json_str = code_block_match.group(1)
            else:
                raise ValueError(f"Could not extract JSON from response: {response_text[:200]}")
        else:
            json_str = json_match.group(0)

        try:
            cards_data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {e}") from e

        if not isinstance(cards_data, list):
            raise ValueError("Response must be a JSON array")

        return cards_data

    def _create_flashcards(
        self, cards_data: list[dict[str, Any]], note: Note
    ) -> list[Flashcard]:
        """Create Flashcard objects from parsed data.

        Args:
            cards_data: List of flashcard dictionaries from LLM
            note: Source note for metadata

        Returns:
            List of Flashcard objects
        """
        flashcards: list[Flashcard] = []
        tags = self._generate_tags(note)

        for card in cards_data:
            front = card.get("front", "").strip()
            back = card.get("back", "").strip()
            card_type_str = card.get("type", "basic").lower()

            if not front or not back:
                continue

            card_type = (
                FlashcardType.CLOZE if card_type_str == "cloze" else FlashcardType.BASIC
            )

            flashcard = Flashcard(
                front=front,
                back=back,
                card_type=card_type,
                note_id=note.id or 0,
                tags=tags,
                deck_name=self.deck_name,
            )
            flashcards.append(flashcard)

        return flashcards

    def _generate_tags(self, note: Note) -> list[str]:
        """Generate Anki tags from note metadata.

        Args:
            note: Source note

        Returns:
            List of tag strings
        """
        tags: list[str] = ["recaller"]

        if note.category:
            # Convert category to valid tag (lowercase, underscores)
            category_tag = note.category.lower().replace(" ", "_")
            tags.append(f"category::{category_tag}")

        if note.source:
            # Add source as tag if it's short enough
            source_clean = note.source.strip()
            if len(source_clean) <= 50 and source_clean:
                source_tag = source_clean.lower().replace(" ", "_")
                # Remove special characters
                source_tag = re.sub(r"[^a-z0-9_:/-]", "", source_tag)
                if source_tag:
                    tags.append(f"source::{source_tag}")

        return tags
