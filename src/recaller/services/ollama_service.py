"""Ollama service for LLM interactions."""

import os
from typing import Any, Optional

from ollama import Client  # type: ignore[import-untyped]
from tenacity import retry, stop_after_attempt, wait_exponential


class OllamaError(Exception):
    """Error from Ollama API."""

    pass


class OllamaService:
    """Service for interacting with Ollama LLM.

    Uses the ollama Python package for API interactions.
    """

    DEFAULT_MODEL = "gpt-oss:120b-cloud"
    DEFAULT_HOST = "https://ollama.com"

    def __init__(
        self,
        model_name: Optional[str] = None,
        host: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize the Ollama service.

        Args:
            model_name: Model to use (default: gpt-oss:120b-cloud)
            host: Ollama API host (default: https://ollama.com)
            api_key: API key for authentication (default: from OLLAMA_API_KEY env var)
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.host = host or self.DEFAULT_HOST
        self.api_key = api_key or os.environ.get("OLLAMA_API_KEY", "")

        # Initialize client with authentication
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        self._client = Client(host=self.host, headers=headers if headers else None)

    @property
    def client(self) -> Client:
        """Get the Ollama client."""
        return self._client

    def check_connection(self) -> bool:
        """Check if Ollama is available.

        Returns:
            True if connected, False otherwise
        """
        try:
            # Try to list models to check connection
            self._client.list()
            return True
        except Exception:
            return False

    def list_models(self) -> list[str]:
        """List available models.

        Returns:
            List of model names
        """
        try:
            response = self._client.list()
            models = response.get("models", [])
            return [m.get("name", "") for m in models if m.get("name")]
        except Exception:
            return []

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        """Generate content using Ollama chat API.

        Args:
            prompt: The prompt to send
            system: Optional system prompt

        Returns:
            Generated text response
        """
        messages: list[dict[str, Any]] = []

        if system:
            messages.append({"role": "system", "content": system})

        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat(
                model=self.model_name,
                messages=messages,
                stream=False,
            )
            return response["message"]["content"].strip()
        except Exception as e:
            error_msg = str(e)
            if "connection" in error_msg.lower():
                raise OllamaError(
                    f"Cannot connect to Ollama at {self.host}. "
                    "Check your connection and API key."
                ) from e
            elif "timeout" in error_msg.lower():
                raise OllamaError("Ollama request timed out") from e
            else:
                raise OllamaError(f"Ollama request failed: {e}") from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def generate_combined_title(self, title_a: str, title_b: str) -> str:
        """Generate a combined title for two similar notes.

        Args:
            title_a: First note title
            title_b: Second note title

        Returns:
            A combined title that represents both notes
        """
        prompt = (
            f'Combine these two note titles into one concise title:\n'
            f'1. "{title_a}"\n'
            f'2. "{title_b}"\n\n'
            f'Return ONLY the combined title, nothing else. '
            f'No quotes, no explanation, just the title.'
        )

        system = (
            "You are a helpful assistant that combines similar note titles. "
            "Always respond with just the combined title, no extra text."
        )

        result = self.generate(prompt, system=system)
        # Clean up the response
        result = result.strip().strip('"').strip("'")
        # Take first line if multiple lines
        if "\n" in result:
            result = result.split("\n")[0].strip()
        return result

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def generate_combined_title_for_group(self, titles: list[str]) -> str:
        """Generate a combined title for a group of similar notes.

        Args:
            titles: List of note titles to combine

        Returns:
            A combined title that represents all notes
        """
        if len(titles) == 1:
            return titles[0]

        if len(titles) == 2:
            return self.generate_combined_title(titles[0], titles[1])

        titles_formatted = "\n".join(f"- {title}" for title in titles)

        prompt = (
            f"Combine these note titles into one concise title:\n"
            f"{titles_formatted}\n\n"
            f"Return ONLY the combined title, nothing else. "
            f"No quotes, no explanation, just the title."
        )

        system = (
            "You are a helpful assistant that combines similar note titles. "
            "Always respond with just the combined title, no extra text."
        )

        result = self.generate(prompt, system=system)
        result = result.strip().strip('"').strip("'")
        if "\n" in result:
            result = result.split("\n")[0].strip()
        return result

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def generate_flashcards_json(
        self,
        title: str,
        category: str,
        source: str,
        content: str,
        min_cards: int = 1,
        max_cards: int = 3,
    ) -> str:
        """Generate flashcards as JSON for a note.

        Args:
            title: Note title
            category: Note category
            source: Note source
            content: Note content
            min_cards: Minimum cards to generate
            max_cards: Maximum cards to generate

        Returns:
            JSON string containing flashcard array
        """
        content_preview = content[:2000] if len(content) > 2000 else content

        prompt = f"""Generate {min_cards}-{max_cards} flashcards from this note.

Title: {title}
Category: {category or "General"}
Source: {source or "Unknown"}

Content:
{content_preview}

Return a JSON array of flashcards. Each flashcard must have:
- "front": the question
- "back": the answer
- "type": "basic" or "cloze"

Example format:
[{{"front": "What is X?", "back": "X is Y", "type": "basic"}}]

Return ONLY the JSON array, no other text."""

        system = (
            "You are an expert educator creating flashcards. "
            "Always respond with valid JSON only, no markdown or explanation."
        )

        return self.generate(prompt, system=system)
