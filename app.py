import streamlit as st
import os
import tempfile
from typing import List, Dict
import time
import matplotlib.pyplot as plt

from utils.pdf_extractor import extract_text_from_pdf
from utils.chunker import split_text_into_chunks
from utils.embeddings import embed_chunks
from utils.vector_store import VectorStore
from utils.mcq_generator import generate_mcqs
from utils.analysis_generator import generate_exam_analysis
from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    FAISS_TOP_K,
    GEMINI_API_KEY,
    EMBEDDING_MODEL,
    LLM_MODEL,
)

st.set_page_config(page_title="PDF MCQ Exam Generator", page_icon="🧠", layout="centered")
st.title("📄 PDF‑Based MCQ Exam Generator & Analyzer")
st.caption("Upload a PDF, generate multiple‑choice questions, answer them, and receive a detailed performance analysis using Gemini.")

import random

# ---------- Helper Functions ----------

def initialize_vector_store(chunks: List[str], embeddings: List[List[float]]) -> VectorStore:
    """Create a FAISS vector store and add embeddings for the given chunks."""
    if not embeddings:
        st.error("No embeddings were generated.")
        st.stop()
    dim = len(embeddings[0])
    store = VectorStore(embedding_dim=dim)
    store.add_embeddings(chunks, embeddings)
    return store

def retrieve_relevant_chunks(store: VectorStore, query: str, k: int = FAISS_TOP_K) -> str:
    """Embed a query string and retrieve k random chunks from the top 3*k most similar chunks.
    
    This ensures variety in the context while keeping relevance.
    """
    query_emb = embed_chunks([query])[0]
    # Retrieve more candidates than needed to allow for random selection
    candidate_k = k * 3 
    results = store.search(query_emb, k=candidate_k)
    
    # results is a list of (chunk_text, distance)
    # Randomly select 'k' chunks from the larger pool of relevant chunks
    num_to_select = min(k, len(results))
    selected_results = random.sample(results, num_to_select)
    
    # Shuffle them to further vary the prompt structure
    random.shuffle(selected_results)
    
    retrieved = " ".join([chunk for chunk, _ in selected_results])
    return retrieved

# ---------- Session State ----------
if "mcqs" not in st.session_state:
    st.session_state.mcqs = []  # List of dicts from Gemini
if "answers" not in st.session_state:
    st.session_state.answers = {}  # question_index -> selected option
if "store" not in st.session_state:
    st.session_state.store = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

# ---------- File Upload ----------
uploaded_file = st.file_uploader("Upload your PDF document", type=["pdf"])


if uploaded_file is not None:
    # Use a safe identifier for the file to prevent re-processing on every interaction
    file_key = f"{uploaded_file.name}_{uploaded_file.size}"
    
    # Only process if it's a new file or hasn't been processed yet
    if "processed_file_key" not in st.session_state or st.session_state.processed_file_key != file_key:
        
        # Reset state for new file
        st.session_state.pdf_processed = False
        st.session_state.mcqs = []
        st.session_state.answers = {}
        st.session_state.quiz_results = []
        st.session_state.current_q_index = 0
        st.session_state.feedback_shown = False
        
        with st.spinner("Processing PDF (Extracting text & Generating embeddings)..."):
            # Save to a temporary location for pdfplumber to read
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            try:
                # Extract text
                raw_text = extract_text_from_pdf(tmp_path)
                
                # Chunking
                chunks = split_text_into_chunks(raw_text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
                
                # Embedding
                embeddings = embed_chunks(chunks)
                
                # Store in session state
                st.session_state.store = initialize_vector_store(chunks, embeddings)
                st.session_state.pdf_processed = True
                st.session_state.processed_file_key = file_key
                st.session_state.preview_text = raw_text[:500] + "..."
                st.session_state.num_chunks = len(chunks)
                
                st.success("PDF processed and embeddings generated successfully!")
            except Exception as e:
                st.error(f"Error processing PDF: {e}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
    else:
        # If already processed, just show a small indicator
        st.success(f"File '{uploaded_file.name}' is ready! ({st.session_state.num_chunks} chunks)")
        with st.expander("View extracted text preview"):
            st.code(st.session_state.get("preview_text", ""))

# ---------- MCQ Generation Controls ----------
if st.session_state.pdf_processed:
    difficulty = st.selectbox("Select exam difficulty", options=["Easy", "Medium", "Hard"], index=0)
    num_mcqs = st.slider("Number of MCQs to generate", min_value=1, max_value=20, value=5)
    generate_btn = st.button("Generate MCQs")
    if generate_btn:
        with st.spinner("Retrieving relevant context and generating MCQs…"):
            # Use the difficulty string as a simple query to fetch relevant chunks
            context = retrieve_relevant_chunks(st.session_state.store, difficulty)
            mcqs = generate_mcqs(context, num_questions=num_mcqs, difficulty=difficulty)
        st.session_state.mcqs = mcqs
        st.session_state.answers = {}
        # Reset quiz state for new batch
        st.session_state.current_q_index = 0
        st.session_state.quiz_results = []
        st.session_state.feedback_shown = False
        st.session_state.q_start_time = time.time()
        st.success(f"Generated {len(mcqs)} MCQs.")

# ---------- MCQ Display & Answer Collection ----------
# ---------- MCQ Display & Answer Collection ----------
if st.session_state.mcqs:
    # Initialize session state for the quiz flow if not present
    if "current_q_index" not in st.session_state:
        st.session_state.current_q_index = 0
    if "quiz_results" not in st.session_state:
        st.session_state.quiz_results = []
    if "q_start_time" not in st.session_state:
        st.session_state.q_start_time = time.time()
    if "feedback_shown" not in st.session_state:
        st.session_state.feedback_shown = False

    current_index = st.session_state.current_q_index
    total_questions = len(st.session_state.mcqs)

    if current_index < total_questions:
        # Progress Bar
        st.progress((current_index) / total_questions)
        st.caption(f"Question {current_index + 1} of {total_questions}")

        mcq = st.session_state.mcqs[current_index]
        st.subheader(f"Q{current_index + 1}: {mcq.get('question', 'No question text')}")
        
        options = mcq.get("options", {})
        option_labels = ["A", "B", "C", "D"]
        
        # Display Radio Button
        selected_opt = st.radio(
            "Select your answer:",
            option_labels,
            format_func=lambda x: f"{x}: {options.get(x, '')}",
            key=f"q_radio_{current_index}",
            disabled=st.session_state.feedback_shown,
            index=None
        )

        # Logic for Submit / Next
        btn_label = "Next Question ➡️" if (current_index + 1 < total_questions) else "Finish Quiz 🏁"
        
        if st.button(btn_label):
            if not selected_opt:
                st.warning("Please select an answer first.")
            else:
                time_taken = time.time() - st.session_state.q_start_time
                st.session_state.quiz_results.append({
                    "question": mcq.get("question"),
                    "options": mcq.get("options"), # Store options to display later
                    "selected": selected_opt,
                    "correct": mcq.get("answer"),
                    "time_taken": time_taken,
                    "topic": mcq.get("topic", "")
                })
                # Move to next question immediately
                st.session_state.current_q_index += 1
                st.session_state.q_start_time = time.time()
                st.rerun()
                    
    else:
        # Final Results Page
        st.balloons()
        st.title("🎉 Quiz Completed!")
        
        results = st.session_state.quiz_results
        total_answered = len(results)
        if total_answered > 0:
            correct_answers = sum([1 for r in results if r['selected'] == r['correct']])
            avg_time = sum([r['time_taken'] for r in results]) / total_answered
            
            # Metrics
            c1, c2, c3 = st.columns(3)
            c1.metric("Score", f"{correct_answers}/{total_answered}")
            c2.metric("Accuracy", f"{(correct_answers/total_answered)*100:.1f}%")
            c3.metric("Avg Time/Question", f"{avg_time:.2f}s")
            
            st.divider()

            # Question-by-Question Review
            st.subheader("🔍 Detailed Review")
            for idx, res in enumerate(results):
                with st.expander(f"Q{idx+1}: {res['question']} ({'✅ Correct' if res['selected'] == res['correct'] else '❌ Incorrect'})"):
                    st.markdown(f"**Your Answer:** {res['selected']} | **Correct Answer:** {res['correct']}")
                    st.caption(f"Time Taken: {res['time_taken']:.2f}s")
                    # Optional: Show options again to give context
                    opts = res.get("options", {})
                    for label, text in opts.items():
                        prefix = "✅ " if label == res['correct'] else ""
                        if label == res['selected'] and label != res['correct']:
                            prefix = "❌ "
                        st.text(f"{prefix}{label}: {text}")

            st.divider()
            
            # Matplotlib Analysis
            st.subheader("📊 Performance Analytics")
            
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                # 1. Time per question (Bar Chart)
                fig1, ax1 = plt.subplots(figsize=(6, 4))
                times = [r['time_taken'] for r in results]
                questions_labels = [f"Q{i+1}" for i in range(total_answered)]
                colors = ['#4CAF50' if r['selected'] == r['correct'] else '#F44336' for r in results]
                
                ax1.bar(questions_labels, times, color=colors)
                ax1.set_ylabel("Time (seconds)")
                ax1.set_title("Time per Question (Green=Correct, Red=Wrong)")
                st.pyplot(fig1)

            with col_chart2:
                # 2. Correct vs Incorrect (Pie Chart)
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                correct_count = correct_answers
                wrong_count = total_answered - correct_answers
                labels = ['Correct', 'Incorrect']
                sizes = [correct_count, wrong_count]
                colors_pie = ['#4CAF50', '#F44336']
                
                ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors_pie)
                ax2.axis('equal') 
                ax2.set_title("Accuracy Breakdown")
                st.pyplot(fig2)

            # Generate Detailed Analysis (LLM)
            # Adapt data for generate_exam_analysis
            user_responses_formatted = [
                {
                    "question": r["question"],
                    "selected_answer": r["selected"],
                    "correct_answer": r["correct"],
                    "topic": r.get("topic", "")
                }
                for r in results
            ]
            
            st.divider()
            st.subheader("📝 Detailed AI Analysis")
            with st.spinner("Generating detailed exam analysis…"):
                analysis_md = generate_exam_analysis(user_responses_formatted)
            st.markdown(analysis_md)
        else:
            st.warning("No questions answered.")

        if st.button("Restart Quiz 🔄"):
            # Reset everything
            st.session_state.current_q_index = 0
            st.session_state.quiz_results = []
            st.session_state.feedback_shown = False
            st.session_state.q_start_time = time.time()
            st.rerun()

# ---------- Footer ----------
st.caption("Powered by Divyansh • Built with Streamlit")
