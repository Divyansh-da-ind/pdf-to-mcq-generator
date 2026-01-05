# pdf-to-mcq-generator
Generate MCQ question papers from PDFs using LLM with exam analysis.


# PDF to MCQ Question Paper Generator (LLM-Based)

This project is an AI-based application that generates Multiple Choice Questions (MCQs) from PDF documents using Large Language Models (LLMs). It allows users to attempt quizzes and receive detailed performance analysis.

---

## Features
- Upload PDF documents
- Extract and chunk text automatically
- Generate MCQs using LLMs
- Select difficulty level (Easy, Medium, Hard)
- Interactive quiz interface
- Score calculation and accuracy tracking
- AI-based performance analysis
- Visual performance charts

---

## Workflow
1. Upload a PDF document  
2. Extract and split text into chunks  
3. Generate embeddings and store them in a vector database  
4. Retrieve relevant content based on difficulty  
5. Generate MCQs using an LLM  
6. Attempt the quiz  
7. View results and detailed analysis  

---

## Tech Stack
- Python
- Streamlit
- Gemini (Embeddings)
- Groq (LLM generation)
- FAISS
- pdfplumber
- langchain
- sentence-transformers
- matplotlib

