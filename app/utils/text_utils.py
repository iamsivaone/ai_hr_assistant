"""Text utilities: cleaning, chunking, and simple heuristics."""

import re
from typing import List


def clean_text(input_text: str) -> str:
    """
    Clean a given text by replacing all occurrences of "\r\n" and "\r" with "\n",
    and replacing all occurrences of one or more whitespace characters with a single space.
    Finally, strip any leading or trailing whitespace from the text.

    Args:
        input_text (str): The text to clean.

    Returns:
        str: The cleaned text.
    """

    # Replace all occurrences of "\r\n" and "\r" with "\n"
    cleaned_text = re.sub(r"[\r\n]", "\n", input_text)

    # Replace all occurrences of one or more whitespace characters with a single space
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)

    # Strip any leading or trailing whitespace from the text
    cleaned_text = cleaned_text.strip()

    return cleaned_text
