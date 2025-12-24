"""Gemini service for LLM interactions."""

from typing import Any, Optional

from google import genai
from tenacity import retry, stop_after_attempt, wait_exponential


class GeminiService:
    """Service for interacting with Google Gemini API.

    Uses Gemini 2.0 Flash-Lite for fast, cost-effective LLM operations.
    """

    DEFAULT_MODEL = "gemini-2.0-flash-lite"

    def __init__(self, api_key: str, model_name: Optional[str] = None):
        """Initialize the Gemini service.

        Args:
            api_key: Google AI API key
            model_name: Model to use (default: gemini-2.0-flash-lite)
        """
        self.api_key = api_key
        self.model_name = model_name or self.DEFAULT_MODEL
        self._client: Any = None

    @property
    def client(self) -> Any:
        """Lazy-load the Gemini client."""
        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _generate(self, prompt: str) -> str:
        """Generate content using the Gemini model.

        Args:
            prompt: The prompt to send to the model

        Returns:
            Generated text response
        """
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return str(response.text).strip().strip('"')

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def generate_combined_title(self, title_a: str, title_b: str) -> str:
        """Generate a combined title for two similar notes.

        Uses Gemini to create a concise title that captures both notes.

        Args:
            title_a: First note title
            title_b: Second note title

        Returns:
            A combined title that represents both notes
        """
        prompt = (
            "You are helping merge two similar notes. "
            "Generate a single, concise title that captures the essence of both notes.\n\n"
            f'Note 1 title: "{title_a}"\n'
            f'Note 2 title: "{title_b}"\n\n'
            "Requirements:\n"
            "- The combined title should be concise (under 100 characters)\n"
            "- It should capture the main topic from both notes\n"
            "- If the notes are about the same thing, use the more descriptive title\n"
            '- Do not add any prefixes like "Combined:" or "Merged:"\n'
            "- Return ONLY the new title, nothing else\n\n"
            "Combined title:"
        )

        return self._generate(prompt)

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
            "You are helping merge several similar notes into one. "
            "Generate a single, concise title that captures the essence of all these notes.\n\n"
            f"Note titles:\n{titles_formatted}\n\n"
            "Requirements:\n"
            "- The combined title should be concise (under 100 characters)\n"
            "- It should capture the main topic shared by all notes\n"
            '- Do not add any prefixes like "Combined:" or "Merged:"\n'
            "- Return ONLY the new title, nothing else\n\n"
            "Combined title:"
        )

        return self._generate(prompt)
