"""File utilities: safe saving of uploads and path helpers."""

import os
from pathlib import Path
from werkzeug.utils import secure_filename

from app.config.constants import ALLOWED_EXTENSIONS




def is_allowed(filename: str) -> bool:
    """
    Checks if a file with the given filename has an allowed extension.

    Args:
        filename (str): Filename to check

    Returns:
        bool: True if the file has an allowed extension, False otherwise
    """
    return filename and filename.split(".")[-1].lower() in ALLOWED_EXTENSIONS


def save_upload(uploaded_file, dest_dir: str) -> str:
    """
    Saves a file uploaded via a Streamlit file uploader to the given destination directory.

    Args:
        uploaded_file: The file to save, as returned by a Streamlit file uploader.
        dest_dir (str): The directory to save the file to.

    Returns:
        str: The full path to the saved file.

    Raises:
        ValueError: If the file has an unsupported extension.
    """
    Path(dest_dir).mkdir(parents=True, exist_ok=True)

    # Clean filename to avoid unsafe characters
    filename = secure_filename(uploaded_file.name)

    if not is_allowed(filename):
        raise ValueError(f"Unsupported file type: {filename}")

    dest_path = Path(dest_dir) / filename

    # Save file
    with open(dest_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return str(dest_path)
