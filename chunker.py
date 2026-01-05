from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List


def split_text_into_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Split raw text into overlapping chunks suitable for embedding.

    Args:
        text: The full extracted text.
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of characters that overlap between consecutive chunks.
    Returns:
        A list of text chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " "],
    )
    return splitter.split_text(text)
