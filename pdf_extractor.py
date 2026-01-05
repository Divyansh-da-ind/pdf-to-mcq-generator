import pdfplumber
from typing import List

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract raw text from a PDF file.

    Args:
        pdf_path: Path to the PDF file.
    Returns:
        A single string containing the concatenated text of all pages.
    """
    all_text: List[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text.append(text)
    return "\n\n".join(all_text)
