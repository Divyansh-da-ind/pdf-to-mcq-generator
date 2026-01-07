import os
from pathlib import Path
from dotenv import load_dotenv

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent

env_locations = [current_dir / ".env", parent_dir / ".env"]
env_loaded = False

for loc in env_locations:
    if loc.exists():
        load_dotenv(dotenv_path=loc)
        env_loaded = True
        break


GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    GEMINI_API_KEY = GEMINI_API_KEY.strip()

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY:
    GROQ_API_KEY = GROQ_API_KEY.strip()

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please set it in .env for embeddings.")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found. Please set it in .env for text generation.")

CHUNK_SIZE: int = 1000  # characters per chunk
CHUNK_OVERLAP: int = 200  # overlapping characters between chunks
FAISS_TOP_K: int = 5

EMBEDDING_MODEL: str = "models/embedding-001"  # Still using Gemini for embeddings
LLM_MODEL: str = "llama-3.3-70b-versatile"  # Updated to supported Groq model

PAGE_TITLE = "PDF‑Based MCQ Exam Generator & Analyzer"
PAGE_ICON = "🧠"

