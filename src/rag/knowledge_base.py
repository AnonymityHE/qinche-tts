"""Build and manage the character knowledge base in ChromaDB."""

from __future__ import annotations

import os
from pathlib import Path

import chromadb

_DEFAULT_KB_DIR = Path(__file__).resolve().parents[2] / "data" / "character_kb"
_DEFAULT_CHROMA_DIR = Path(__file__).resolve().parents[2] / "chroma_db"
_COLLECTION_NAME = "qinche_character"


def _get_embedding_fn():
    """Lazy-load bge-large-zh embedding function."""
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    cache_dir = os.environ.get(
        "BGE_MODEL_PATH",
        str(Path(__file__).resolve().parents[2] / "models" / "bge-large-zh"),
    )
    return SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-large-zh-v1.5",
        cache_folder=cache_dir,
    )


def build_knowledge_base(
    kb_dir: str | Path = _DEFAULT_KB_DIR,
    chroma_dir: str | Path = _DEFAULT_CHROMA_DIR,
    chunk_size: int = 400,
    chunk_overlap: int = 80,
) -> chromadb.Collection:
    """Read markdown files from *kb_dir*, chunk them, and upsert into ChromaDB."""

    kb_dir = Path(kb_dir)
    chroma_dir = Path(chroma_dir)

    documents: list[str] = []
    metadatas: list[dict] = []
    ids: list[str] = []

    for md_file in sorted(kb_dir.glob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        dimension = md_file.stem  # e.g. personality, relationships

        chunks = _split_text(text, chunk_size, chunk_overlap)
        for i, chunk in enumerate(chunks):
            doc_id = f"{dimension}_{i:03d}"
            documents.append(chunk)
            metadatas.append({"dimension": dimension, "source": md_file.name})
            ids.append(doc_id)

    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_or_create_collection(
        name=_COLLECTION_NAME,
        embedding_function=_get_embedding_fn(),
    )

    if documents:
        collection.upsert(documents=documents, metadatas=metadatas, ids=ids)

    return collection


def _split_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks by character count, respecting line boundaries."""
    lines = text.splitlines(keepends=True)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for line in lines:
        current.append(line)
        current_len += len(line)
        if current_len >= size:
            chunks.append("".join(current))
            keep = []
            keep_len = 0
            for prev_line in reversed(current):
                if keep_len + len(prev_line) > overlap:
                    break
                keep.insert(0, prev_line)
                keep_len += len(prev_line)
            current = keep
            current_len = keep_len

    if current:
        chunks.append("".join(current))

    return chunks
