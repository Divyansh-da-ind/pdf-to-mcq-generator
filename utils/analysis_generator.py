import json
from typing import List, Dict
import os
from openai import OpenAI
from config import GROQ_API_KEY, LLM_MODEL

# Prompt template for exam analysis
ANALYSIS_PROMPT_TEMPLATE = """
User answered the following MCQs (provide question, selected answer, correct answer, topic). Summarize performance:
- Total score and percentage
- Weak topics (topics with < 50% correct)
- Common conceptual mistakes
- Personalized improvement suggestions
- Study strategy for weak topics
Return a markdown formatted report.
"""

def generate_exam_analysis(user_responses: List[Dict]) -> str:
    """Generate a detailed exam analysis using Groq (via OpenAI client).

    Args:
        user_responses: List of dictionaries, each containing the keys
            ``question``, ``selected_answer``, ``correct_answer``, and ``topic``.
    Returns:
        A markdown string containing the analysis report.
    """
    # Prepare the input for the model
    responses_json = json.dumps(user_responses, indent=2)
    prompt = f"{ANALYSIS_PROMPT_TEMPLATE}\nUser responses:\n{responses_json}"
    
    client = OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )

    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that analyzes exam performance."
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=LLM_MODEL,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Failed to generate analysis with Groq: {e}"
