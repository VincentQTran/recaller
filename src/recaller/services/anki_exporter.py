"""Anki exporter service using AnkiConnect."""

from dataclasses import dataclass
from typing import Any, Optional

import requests  # type: ignore[import-untyped]
from tenacity import retry, stop_after_attempt, wait_exponential

from recaller.models.flashcard import ExportStatus, Flashcard, FlashcardType


@dataclass
class ExportResult:
    """Result of an export operation."""

    flashcard: Flashcard
    success: bool
    anki_note_id: Optional[int] = None
    error: Optional[str] = None


class AnkiConnectError(Exception):
    """Error from AnkiConnect API."""

    pass


class AnkiExporter:
    """Service for exporting flashcards to Anki via AnkiConnect.

    Requires Anki desktop to be running with AnkiConnect add-on installed.
    """

    ANKICONNECT_VERSION = 6

    def __init__(self, url: str = "http://localhost:8765"):
        """Initialize the Anki exporter.

        Args:
            url: AnkiConnect URL (default: http://localhost:8765)
        """
        self.url = url

    def _invoke(self, action: str, **params: Any) -> Any:
        """Invoke an AnkiConnect action.

        Args:
            action: The AnkiConnect action name
            **params: Parameters for the action

        Returns:
            The result from AnkiConnect

        Raises:
            AnkiConnectError: If the request fails or returns an error
        """
        payload = {
            "action": action,
            "version": self.ANKICONNECT_VERSION,
        }
        if params:
            payload["params"] = params

        try:
            response = requests.post(self.url, json=payload, timeout=30)
            response.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            raise AnkiConnectError(
                "Cannot connect to AnkiConnect. Is Anki running with AnkiConnect add-on?"
            ) from e
        except requests.exceptions.Timeout as e:
            raise AnkiConnectError("AnkiConnect request timed out") from e
        except requests.exceptions.RequestException as e:
            raise AnkiConnectError(f"AnkiConnect request failed: {e}") from e

        result = response.json()

        if result.get("error"):
            raise AnkiConnectError(result["error"])

        return result.get("result")

    def check_connection(self) -> bool:
        """Check if AnkiConnect is available.

        Returns:
            True if connected, False otherwise
        """
        try:
            version = self._invoke("version")
            return version is not None
        except AnkiConnectError:
            return False

    def get_version(self) -> Optional[int]:
        """Get AnkiConnect version.

        Returns:
            Version number or None if not connected
        """
        try:
            result = self._invoke("version")
            return int(result) if result is not None else None
        except AnkiConnectError:
            return None

    def get_deck_names(self) -> list[str]:
        """Get list of all deck names.

        Returns:
            List of deck names
        """
        return self._invoke("deckNames") or []

    def ensure_deck_exists(self, deck_name: str) -> bool:
        """Ensure a deck exists, creating it if necessary.

        Args:
            deck_name: Name of the deck (supports :: for nested decks)

        Returns:
            True if deck exists or was created
        """
        existing_decks = self.get_deck_names()
        if deck_name in existing_decks:
            return True

        # Create the deck
        try:
            self._invoke("createDeck", deck=deck_name)
            return True
        except AnkiConnectError:
            return False

    def _flashcard_to_anki_note(self, flashcard: Flashcard) -> dict[str, Any]:
        """Convert a Flashcard to AnkiConnect note format.

        Args:
            flashcard: The flashcard to convert

        Returns:
            AnkiConnect note dictionary
        """
        # Choose model based on card type
        if flashcard.card_type == FlashcardType.CLOZE:
            return {
                "deckName": flashcard.deck_name,
                "modelName": "Cloze",
                "fields": {
                    "Text": flashcard.front,
                    "Extra": flashcard.back,
                },
                "tags": flashcard.tags,
                "options": {
                    "allowDuplicate": False,
                    "duplicateScope": "deck",
                },
            }
        else:
            return {
                "deckName": flashcard.deck_name,
                "modelName": "Basic",
                "fields": {
                    "Front": flashcard.front,
                    "Back": flashcard.back,
                },
                "tags": flashcard.tags,
                "options": {
                    "allowDuplicate": False,
                    "duplicateScope": "deck",
                },
            }

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
    )
    def add_note(self, flashcard: Flashcard) -> ExportResult:
        """Add a single flashcard to Anki.

        Args:
            flashcard: The flashcard to add

        Returns:
            ExportResult with success status and anki_note_id
        """
        note = self._flashcard_to_anki_note(flashcard)

        try:
            note_id = self._invoke("addNote", note=note)

            if note_id:
                flashcard.anki_note_id = note_id
                flashcard.export_status = ExportStatus.EXPORTED
                return ExportResult(
                    flashcard=flashcard,
                    success=True,
                    anki_note_id=note_id,
                )
            else:
                flashcard.export_status = ExportStatus.FAILED
                return ExportResult(
                    flashcard=flashcard,
                    success=False,
                    error="No note ID returned",
                )

        except AnkiConnectError as e:
            flashcard.export_status = ExportStatus.FAILED
            return ExportResult(
                flashcard=flashcard,
                success=False,
                error=str(e),
            )

    def add_notes(self, flashcards: list[Flashcard]) -> list[ExportResult]:
        """Add multiple flashcards to Anki in batch.

        Args:
            flashcards: List of flashcards to add

        Returns:
            List of ExportResult objects
        """
        if not flashcards:
            return []

        # Ensure all decks exist first
        deck_names = set(f.deck_name for f in flashcards)
        for deck_name in deck_names:
            self.ensure_deck_exists(deck_name)

        # Convert flashcards to AnkiConnect format
        notes = [self._flashcard_to_anki_note(f) for f in flashcards]

        try:
            # Use addNotes for batch operation
            note_ids = self._invoke("addNotes", notes=notes)

            results = []
            for flashcard, note_id in zip(flashcards, note_ids):
                if note_id:
                    flashcard.anki_note_id = note_id
                    flashcard.export_status = ExportStatus.EXPORTED
                    results.append(ExportResult(
                        flashcard=flashcard,
                        success=True,
                        anki_note_id=note_id,
                    ))
                else:
                    flashcard.export_status = ExportStatus.FAILED
                    results.append(ExportResult(
                        flashcard=flashcard,
                        success=False,
                        error="Duplicate or invalid note",
                    ))

            return results

        except AnkiConnectError as e:
            # If batch fails, mark all as failed
            return [
                ExportResult(
                    flashcard=f,
                    success=False,
                    error=str(e),
                )
                for f in flashcards
            ]

    def sync(self) -> bool:
        """Trigger Anki sync with AnkiWeb.

        Returns:
            True if sync was triggered successfully
        """
        try:
            self._invoke("sync")
            return True
        except AnkiConnectError:
            return False

    def get_num_cards_today(self) -> dict[str, int]:
        """Get number of cards reviewed today.

        Returns:
            Dictionary with 'new', 'learning', 'review' counts
        """
        try:
            return self._invoke("getNumCardsReviewedToday") or {}
        except AnkiConnectError:
            return {}
