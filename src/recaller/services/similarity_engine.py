"""Similarity engine for finding similar notes."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from recaller.models.note import Note


@dataclass
class SimilarityPair:
    """Represents a pair of similar notes."""

    note_a: Note
    note_b: Note
    similarity: float

    def __str__(self) -> str:
        return f"{self.note_a.title} <-> {self.note_b.title} ({self.similarity:.2f})"


class SimilarityEngine:
    """Engine for finding similar notes using cosine similarity.

    Compares note embeddings to find pairs above a given threshold.
    """

    def __init__(self, threshold: float = 0.78):
        """Initialize the similarity engine.

        Args:
            threshold: Minimum cosine similarity for notes to be considered similar.
                      Default is 0.78. Range: 0.5 (loose) to 0.95 (strict).
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")
        self.threshold = threshold

    def cosine_similarity(
        self, embedding_a: np.ndarray, embedding_b: np.ndarray
    ) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            embedding_a: First embedding vector
            embedding_b: Second embedding vector

        Returns:
            Cosine similarity score between -1 and 1
        """
        # Normalize vectors
        norm_a = np.linalg.norm(embedding_a)
        norm_b = np.linalg.norm(embedding_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return float(np.dot(embedding_a, embedding_b) / (norm_a * norm_b))

    def find_similar_pairs(
        self,
        notes: list[Note],
        exclude_pairs: Optional[set[tuple[int, int]]] = None,
        same_category_only: bool = True,
    ) -> list[SimilarityPair]:
        """Find all pairs of notes with similarity above threshold.

        Args:
            notes: List of notes with embeddings
            exclude_pairs: Set of (note_id, note_id) tuples to exclude from comparison
            same_category_only: If True, only compare notes with the same category

        Returns:
            List of SimilarityPair objects, sorted by similarity (descending)
        """
        if exclude_pairs is None:
            exclude_pairs = set()

        # Filter notes that have embeddings
        notes_with_embeddings = [n for n in notes if n.embedding is not None]

        if len(notes_with_embeddings) < 2:
            return []

        pairs: list[SimilarityPair] = []

        # Compare all pairs
        for i, note_a in enumerate(notes_with_embeddings):
            for note_b in notes_with_embeddings[i + 1 :]:
                # Skip if categories don't match (when same_category_only is True)
                if same_category_only:
                    # Both must have a category and they must match
                    if not note_a.category or not note_b.category:
                        continue
                    if note_a.category.lower() != note_b.category.lower():
                        continue

                # Skip if this pair should be excluded
                if note_a.id and note_b.id:
                    pair_key = tuple(sorted([note_a.id, note_b.id]))
                    if pair_key in exclude_pairs:
                        continue

                # Compute similarity
                similarity = self.cosine_similarity(
                    note_a.embedding,  # type: ignore
                    note_b.embedding,  # type: ignore
                )

                if similarity >= self.threshold:
                    pairs.append(SimilarityPair(
                        note_a=note_a,
                        note_b=note_b,
                        similarity=similarity,
                    ))

        # Sort by similarity (highest first)
        pairs.sort(key=lambda p: p.similarity, reverse=True)

        return pairs

    def find_similar_to_note(
        self,
        target_note: Note,
        candidate_notes: list[Note],
    ) -> list[SimilarityPair]:
        """Find notes similar to a specific target note.

        Args:
            target_note: The note to find similar notes for (must have embedding)
            candidate_notes: List of notes to compare against

        Returns:
            List of SimilarityPair objects, sorted by similarity (descending)
        """
        if target_note.embedding is None:
            return []

        pairs: list[SimilarityPair] = []

        for candidate in candidate_notes:
            # Skip the target note itself
            if candidate.notion_page_id == target_note.notion_page_id:
                continue

            if candidate.embedding is None:
                continue

            similarity = self.cosine_similarity(
                target_note.embedding,
                candidate.embedding,
            )

            if similarity >= self.threshold:
                pairs.append(SimilarityPair(
                    note_a=target_note,
                    note_b=candidate,
                    similarity=similarity,
                ))

        pairs.sort(key=lambda p: p.similarity, reverse=True)
        return pairs

    def compute_similarity_matrix(
        self, notes: list[Note]
    ) -> tuple[np.ndarray, list[Note]]:
        """Compute a full similarity matrix for all notes.

        Args:
            notes: List of notes with embeddings

        Returns:
            Tuple of (similarity_matrix, filtered_notes)
            - similarity_matrix: NxN matrix of cosine similarities
            - filtered_notes: Notes that have embeddings (in same order as matrix)
        """
        # Filter notes with embeddings
        filtered = [n for n in notes if n.embedding is not None]

        if len(filtered) < 2:
            return np.array([]), filtered

        # Stack embeddings into matrix
        embeddings = np.stack([n.embedding for n in filtered])  # type: ignore

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        normalized = embeddings / norms

        # Compute similarity matrix (dot product of normalized vectors)
        similarity_matrix = np.dot(normalized, normalized.T)

        return similarity_matrix, filtered

    def group_similar_notes(
        self, pairs: list[SimilarityPair]
    ) -> list[list[Note]]:
        """Group related notes using connected components.

        Notes that are transitively similar are grouped together.

        Args:
            pairs: List of similar note pairs

        Returns:
            List of note groups (each group is a list of similar notes)
        """
        if not pairs:
            return []

        # Build adjacency using note IDs
        adjacency: dict[str, set[str]] = {}

        for pair in pairs:
            id_a = pair.note_a.notion_page_id
            id_b = pair.note_b.notion_page_id

            if id_a not in adjacency:
                adjacency[id_a] = set()
            if id_b not in adjacency:
                adjacency[id_b] = set()

            adjacency[id_a].add(id_b)
            adjacency[id_b].add(id_a)

        # Build note lookup
        note_lookup: dict[str, Note] = {}
        for pair in pairs:
            note_lookup[pair.note_a.notion_page_id] = pair.note_a
            note_lookup[pair.note_b.notion_page_id] = pair.note_b

        # Find connected components using BFS
        visited: set[str] = set()
        groups: list[list[Note]] = []

        for start_id in adjacency:
            if start_id in visited:
                continue

            # BFS to find all connected notes
            group_ids: list[str] = []
            queue = [start_id]

            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue

                visited.add(current)
                group_ids.append(current)

                for neighbor in adjacency.get(current, []):
                    if neighbor not in visited:
                        queue.append(neighbor)

            # Convert IDs to notes
            group = [note_lookup[nid] for nid in group_ids]
            groups.append(group)

        return groups
