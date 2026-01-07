from typing import List
from sentence_transformers import SentenceTransformer

import streamlit as st

# Load a small, fast, and high-quality local model
# This runs on your CPU and does not require an API key or internet after download.
# 'all-MiniLM-L6-v2' is a standard choice for RAG (384 dimensions).
# Load a small, fast, and high-quality local model
# This runs on your CPU and does not require an API key or internet after download.
# 'all-MiniLM-L6-v2' is a standard choice for RAG (384 dimensions).

try:
    # Try using Streamlit cache if available
    import streamlit as st
    # Check if we're actually running inside Streamlit by looking for an active script run context
    # or just rely on st.cache_resource not crashing (it might just warn)
    # A simple way:
    @st.cache_resource
    def load_model():
        return SentenceTransformer('all-MiniLM-L6-v2')
except Exception:
    # Fallback for FastAPI or other contexts
    def load_model():
        return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

def get_embedding(text: str) -> List[float]:
    """Return the embedding vector for a given text using local HuggingFace model.

    Args:
        text: The input string to embed.
    Returns:
        A list of floats representing the embedding.
    """
    # encode returns a numpy array, convert to list for compatibility
    return model.encode(text).tolist()

def embed_chunks(chunks: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of text chunks.

    Args:
        chunks: List of text strings.
    Returns:
        List of embedding vectors corresponding to each chunk.
    """
    # SentenceTransformer can encode a list of strings efficiently in a batch
    embeddings = model.encode(chunks)
    return embeddings.tolist()
