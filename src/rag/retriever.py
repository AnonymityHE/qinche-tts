"""Retrieve relevant character context from the knowledge base."""

from __future__ import annotations

from pathlib import Path

import chromadb

from .knowledge_base import _COLLECTION_NAME, _DEFAULT_CHROMA_DIR, _get_embedding_fn


class CharacterRetriever:
    """Query ChromaDB for character context relevant to a scene description."""

    def __init__(self, chroma_dir: str | Path = _DEFAULT_CHROMA_DIR, top_k: int = 3):
        self.top_k = top_k
        self._client = chromadb.PersistentClient(path=str(chroma_dir))
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            embedding_function=_get_embedding_fn(),
        )

    def retrieve(self, scene_description: str) -> list[str]:
        """Return top-k character context fragments for the given scene."""
        results = self._collection.query(
            query_texts=[scene_description],
            n_results=self.top_k,
        )
        documents = results.get("documents", [[]])[0]
        return documents
