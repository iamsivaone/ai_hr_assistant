"""
Global configuration for the AI-Powered HR Assistant.

Loads environment variables, defines default directories,
and exposes global settings via the `settings` object.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import logging

load_dotenv()


class Settings:
    """Holds all configurable environment-dependent settings."""

    def __init__(self):
        # API / LLM configs
        self.llm_key = os.getenv("GROQ_API_KEY", "")
        self.llm_provider = os.getenv("LLM_PROVIDER", "groq")
        self.llm_model = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
        self.llm_temperature = float(os.getenv("LLM_TEMPERATURE", 0.0))

        # Embedding model
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

        # Directories
        self.base_dir = Path(__file__).resolve().parents[2]
        self.upload_dir = Path(os.getenv("UPLOAD_DIR", "./data/uploads"))
        self.index_dir = Path(
            os.getenv("FAISS_INDEX_PATH", "./data/indexes/faiss_index")
        )

        # Vectorstore type
        self.vectorstore = os.getenv("VECTORSTORE", "faiss")

        # Chunking for LLM
        self.max_chunk = int(os.getenv("MAX_CHUNK", 1000))
        self.chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 100))

        # Logging
        log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
        self.log_level = getattr(logging, log_level_str, logging.INFO)

        # Ensure folders exist
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Prompts file
        self.prompts_file = Path(os.getenv("PROMPTS_FILE", "./app/config/prompts.yaml"))


        # Sanity check
        if not self.llm_key:
            print("⚠️  Warning: GROQ_API_KEY not set in environment!")


# Instantiate a global settings object
settings = Settings()
