from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import shutil
import os
import tempfile
from typing import List, Dict, Optional
import json

from utils.pdf_extractor import extract_text_from_pdf
from utils.chunker import split_text_into_chunks
from utils.embeddings import embed_chunks
from utils.vector_store import VectorStore
from utils.mcq_generator import generate_mcqs
from utils.analysis_generator import generate_exam_analysis
from config import CHUNK_SIZE, CHUNK_OVERLAP, FAISS_TOP_K

app = FastAPI()

# Global state (simple in-memory for this demo)
# In production, use Redis or a proper database
class AppState:
    def __init__(self):
        self.raw_text = ""
        self.chunks = []
        self.vector_store = None

state = AppState()

class GenerateRequest(BaseModel):
    difficulty: str
    num_questions: int

class AnalysisRequest(BaseModel):
    user_responses: List[Dict]

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        
        raw_text = extract_text_from_pdf(tmp_path)
        chunks = split_text_into_chunks(raw_text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        embeddings = embed_chunks(chunks)
    
        store = VectorStore(embedding_dim=len(embeddings[0]))
        store.add_embeddings(chunks, embeddings)
        
        state.raw_text = raw_text
        state.chunks = chunks
        state.vector_store = store
        
     
        os.remove(tmp_path)
        
        return {"message": "PDF processed successfully", "chunks_count": len(chunks)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate")
async def generate_quiz(req: GenerateRequest):
    if not state.vector_store:
        raise HTTPException(status_code=400, detail=u"No PDF processed yet.")
    
    from utils.embeddings import model
    
    query_emb = embed_chunks([req.difficulty])[0]
    results = state.vector_store.search(query_emb, k=FAISS_TOP_K)
    context_text = " ".join([chunk for chunk, _ in results])
    
    mcqs = generate_mcqs(context_text, num_questions=req.num_questions, difficulty=req.difficulty)
    return mcqs

@app.post("/api/analyze")
async def analyze_results(req: AnalysisRequest):
    analysis_md = generate_exam_analysis(req.user_responses)
    return {"analysis": analysis_md}


static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

