import json
from typing import List, Dict
import os
from openai import OpenAI
from config import GROQ_API_KEY, LLM_MODEL

# Prompt template for MCQ generation
MCQ_PROMPT_TEMPLATE = '''
You are given the following context extracted from a PDF document:
"""
{retrieved_chunks}
"""
Generate {num_questions} multiple‑choice questions of {difficulty} difficulty strictly based on the above content. For each question, provide:
- "question": string
- "options": {{"A": ..., "B": ..., "C": ..., "D": ...}}
- "answer": "A"|"B"|"C"|"D"
- "topic": short topic name extracted from the content.
Return a JSON array of objects.
'''

def generate_mcqs(retrieved_chunks: str, num_questions: int, difficulty: str) -> List[Dict]:
    """Generate multiple‑choice questions using Groq (via OpenAI client).

    Args:
        retrieved_chunks: Concatenated text chunks relevant to the query.
        num_questions: Number of MCQs to generate.
        difficulty: Desired difficulty level (Easy, Medium, Hard).
    Returns:
        A list of dictionaries, each representing an MCQ.
    """
    prompt = MCQ_PROMPT_TEMPLATE.format(
        retrieved_chunks=retrieved_chunks,
        num_questions=num_questions,
        difficulty=difficulty,
    )
    
    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )

    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that generates multiple-choice questions in JSON format."
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=LLM_MODEL,
            response_format={"type": "json_object"},
            temperature=0.7,
        )
        content = response.choices[0].message.content
        # Ensure the content is parsed as a list. Sometimes models wrap it in a key like "questions".
        parsed_json = json.loads(content)
        
        # Handle case where model returns {"questions": [...]} instead of [...]
        if isinstance(parsed_json, dict):
            # Look for a list value in the dictionary
            for key, value in parsed_json.items():
                if isinstance(value, list):
                    return value
            # If no list found, wrap the dict in a list (fallback)
            return [parsed_json]
            
        if not isinstance(parsed_json, list):
             raise ValueError("Model output is not a JSON list.")
             
        return parsed_json

    except Exception as e:
        raise ValueError(f"Failed to generate MCQs with Groq: {e}")
