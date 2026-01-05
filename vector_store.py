import os
import faiss
import numpy as np
from typing import List, Tuple

class VectorStore:
    """FAISS vector store for storing and retrieving text chunk embeddings.

    This simple wrapper creates an IndexFlatL2 index (L2 distance) and keeps a
    parallel list of the original text chunks so that a search can return the
    most relevant chunks.
    """

    def __init__(self, embedding_dim: int):
        self.dimension = embedding_dim
        # IndexFlatL2 is a simple, non‑quantized index suitable for small to medium data.
        self.index = faiss.IndexFlatL2(self.dimension)
        self.id_to_chunk: List[str] = []

    def add_embeddings(self, chunks: List[str], embeddings: List[List[float]]) -> None:
        """Add a batch of embeddings and their corresponding text chunks.

        Args:
            chunks: List of raw text chunks.
            embeddings: List of embedding vectors (same order as ``chunks``).
        """
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        # Convert to numpy array of shape (n_vectors, dim) with dtype float32
        vectors = np.array(embeddings, dtype=np.float32)
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension mismatch: expected {self.dimension}, got {vectors.shape[1]}")
        self.index.add(vectors)
        self.id_to_chunk.extend(chunks)

    def search(self, query_embedding: List[float], k: int = 5) -> List[Tuple[str, float]]:
        """Search for the *k* most similar chunks to the query embedding.

        Returns a list of tuples ``(chunk_text, distance)`` sorted by increasing
        distance (i.e., most similar first).
        """
        query_vec = np.array([query_embedding], dtype=np.float32)
        distances, indices = self.index.search(query_vec, k)
        results: List[Tuple[str, float]] = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < 0 or idx >= len(self.id_to_chunk):
                continue
            results.append((self.id_to_chunk[idx], float(dist)))
        return results
