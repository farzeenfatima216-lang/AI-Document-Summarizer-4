
"""Embedding and vector search utilities for document question answering."""

import hashlib
from typing import Any, Sequence


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_EMBEDDING_MODEL = None


class RAGUnavailableError(RuntimeError):
    """Raised when the optional RAG dependencies or model are unavailable."""


class FAISSRetriever:
    """Create embeddings and retrieve the nearest document chunks."""

    def __init__(self, chunks: Sequence[Any], embedding_model: Any = None):
        self.records = [
            chunk if isinstance(chunk, dict) else {"text": chunk}
            for chunk in chunks
            if (chunk.get("text") if isinstance(chunk, dict) else chunk)
        ]
        self.chunks = [record["text"] for record in self.records]
        if not self.chunks:
            raise ValueError("At least one non-empty chunk is required")

        try:
            import faiss
        except ImportError as exc:
            raise RAGUnavailableError(
                "RAG requires faiss-cpu and sentence-transformers. "
                "Install requirements.txt."
            ) from exc

        if embedding_model is None:
            embedding_model = get_embedding_model()

        self.embedding_model = embedding_model
        embeddings = self._encode(self.chunks)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
        self.embedding_dimension = embeddings.shape[1]

    def _encode(self, texts: Sequence[str]):
        import numpy as np

        embeddings = self.embedding_model.encode(
            list(texts), convert_to_numpy=True, show_progress_bar=False
        )
        embeddings = np.asarray(embeddings, dtype="float32")
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(norms, 1e-12)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        if not query or not query.strip():
            return []
        limit = min(max(top_k, 1), len(self.chunks))
        scores, indices = self.index.search(self._encode([query]), limit)
        return [
            {
                "chunk": self.chunks[index],
                "score": float(score),
                "index": int(index),
                "text": self.chunks[index],
                "page_number": self.records[index].get("page_number"),
                "paragraph_number": self.records[index].get("paragraph_number"),
                "section": self.records[index].get("section", "Document"),
                "file_name": self.records[index].get("file_name"),
                "document_name": self.records[index].get("document_name"),
            }
            for score, index in zip(scores[0], indices[0])
            if index >= 0
        ]


_RETRIEVER_CACHE: dict[str, FAISSRetriever] = {}


def get_cached_retriever(chunks: Sequence[Any]) -> FAISSRetriever:
    """Reuse an index for repeated questions against one document."""
    cache_key = hashlib.sha256(
        "\x00".join(
            (chunk.get("text", "") if isinstance(chunk, dict) else chunk) or ""
            for chunk in chunks
        ).encode("utf-8")
    ).hexdigest()
    retriever = _RETRIEVER_CACHE.get(cache_key)
    if retriever is None:
        retriever = FAISSRetriever(chunks)
        _RETRIEVER_CACHE[cache_key] = retriever
    return retriever


def get_embedding_model():
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is not None:
        return _EMBEDDING_MODEL

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RAGUnavailableError(
            "RAG requires faiss-cpu and sentence-transformers. "
            "Install requirements.txt."
        ) from exc

    try:
        _EMBEDDING_MODEL = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
    except Exception as exc:  # pragma: no cover - local model/runtime
        raise RAGUnavailableError(
            "Could not load embedding model "
            f"{DEFAULT_EMBEDDING_MODEL}: {exc}"
        ) from exc
    return _EMBEDDING_MODEL


def retrieve_relevant_chunks(
    query: str, chunks: Sequence[Any], top_k: int = 3
) -> list[dict[str, Any]]:
    """Return semantically relevant chunks, or raise a clear setup error."""
    return get_cached_retriever(chunks).retrieve(
        query, top_k=top_k
    )
