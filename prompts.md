# Prompt Templates for MCQ Generation and Exam Analysis

## MCQ Generation Prompt
```
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
```

## Exam Analysis Prompt
```
User answered the following MCQs (provide question, selected answer, correct answer, topic). Summarize performance:
- Total score and percentage
- Weak topics (topics with < 50% correct)
- Common conceptual mistakes
- Personalized improvement suggestions
- Study strategy for weak topics
Return a markdown formatted report.
```
