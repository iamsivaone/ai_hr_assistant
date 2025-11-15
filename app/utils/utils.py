import yaml
from pathlib import Path
import pdfplumber
import docx
import re
from typing import Dict

from app.utils.text_utils import clean_text
from app.config import settings


def load_prompt(key: str) -> str:
    """
    Loads a prompt from the configuration file using the given key.

    Args:
        key (str): Key to retrieve the prompt from the configuration file.

    Returns:
        str: The loaded prompt. If no prompt is found with the given key, an empty string is returned.
    """
    with open(settings.prompts_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get(key, "")


def extract_text_from_pdf(path: str) -> str:
    """
    Extracts text from a PDF file using pdfplumber.

    Args:
        path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF file.
    """
    text = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text.append(page_text)
    return "\n".join(text)


def extract_text_from_docx(path: str) -> str:
    """
    Extracts text from a DOCX file using python-docx.

    Args:
        path (str): Path to the DOCX file.

    Returns:
        str: Extracted text from the DOCX file.
    """
    doc = docx.Document(path)
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(paragraphs)


def extract_text(path: str) -> str:
    """
    Extracts text from a PDF/DOCX/TXT file.

    Args:
        path (str): Path to the file.

    Returns:
        str: Extracted text from the file.

    Raises:
        ValueError: If the file type is not supported.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        raw = extract_text_from_pdf(path)
    elif suffix == ".docx":
        raw = extract_text_from_docx(path)
    elif suffix == ".txt":
        raw = p.read_text(encoding="utf-8", errors="ignore")
    else:
        raise ValueError("Unsupported file type")
    return clean_text(raw)
