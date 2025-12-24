"""Embedding service for generating note embeddings."""

from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from recaller.models.note import Note


class EmbeddingService:
    """Service for generating embeddings from note content.

    Uses sentence-transformers to generate 384-dimensional embeddings
    that can be used for semantic similarity comparisons.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedding service.

        Args:
            model_name: Name of the sentence-transformers model to use.
                       Default is 'all-MiniLM-L6-v2' (384 dimensions, fast).
        """
        self.model_name = model_name
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the model on first use."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text.

        Args:
            text: Text to generate embedding for

        Returns:
            384-dimensional numpy array (float32)
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.astype(np.float32)

    def generate_embeddings(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings for multiple texts (batch).

        Args:
            texts: List of texts to generate embeddings for

        Returns:
            List of 384-dimensional numpy arrays
        """
        if not texts:
            return []

        embeddings = self.model.encode(texts, convert_to_numpy=True, batch_size=32)
        return [emb.astype(np.float32) for emb in embeddings]

    def generate_note_embedding(self, note: Note) -> np.ndarray:
        """Generate embedding for a note using its title.

        Args:
            note: Note to generate embedding for

        Returns:
            384-dimensional numpy array
        """
        # Use title for embedding (could be extended to include content)
        return self.generate_embedding(note.title)

    def generate_note_embeddings(self, notes: list[Note]) -> dict[int, np.ndarray]:
        """Generate embeddings for multiple notes.

        Args:
            notes: List of notes to generate embeddings for

        Returns:
            Dict mapping note.id to embedding array.
            Notes without an id are skipped.
        """
        # Filter notes that have an id
        notes_with_id = [n for n in notes if n.id is not None]

        if not notes_with_id:
            return {}

        # Extract titles
        titles = [n.title for n in notes_with_id]

        # Generate embeddings in batch
        embeddings = self.generate_embeddings(titles)

        # Map to note ids
        return {
            note.id: emb
            for note, emb in zip(notes_with_id, embeddings)
            if note.id is not None
        }

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension for the current model."""
        return self.model.get_sentence_embedding_dimension()
