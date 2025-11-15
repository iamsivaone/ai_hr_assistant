import numpy as np
import math
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from app.utils.llm_utils import get_llm
from app.utils.text_utils import clean_text

from app.utils.utils import load_prompt


class ResumeMatcher:
    def __init__(self, jd_text: str, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize a ResumeMatcher instance.

        Args:
            jd_text (str): Raw job description text.
            model_name (str, optional): SentenceTransformer model name. Defaults to "all-MiniLM-L6-v2".

        Attributes:
            jd_text (str): Cleaned job description text.
            model (SentenceTransformer): SentenceTransformer instance.
            jdembedding (np.ndarray): Precomputed JD embedding.
            llm (Optional[LLM]): Groq LLM instance for semantic search.
        """
        print(
            f"Initializing ResumeMatcher with jd_text {jd_text} and model_name {model_name}"
        )
        self.jd_text = clean_text(jd_text)
        print(f"Cleaned JD text: {self.jd_text}")
        self.model = SentenceTransformer(model_name)
        print(f"Initialized SentenceTransformer model {model_name}")
        # Precompute JD embedding once
        self.jd_embedding = self.model.encode([self.jd_text], convert_to_numpy=True)[0]
        print(f"Precomputed JD embedding shape: {self.jd_embedding.shape}")

        # Initialize Groq LLM (same as summarizer)
        try:
            self.llm = get_llm()
            print("Initialized Groq LLM instance successfully")
        except Exception as e:
            print(f"Failed to initialize LLM for matching: {e}")
            self.llm = None

    def nlp_score(self, resume_text: str, skill_weight: float = 0.6) -> float:
        """
        Compute an NLP-based score for a resume against a job description.

        The score is a weighted combination of semantic similarity and keyword overlap.

        Args:
            resume_text (str): Raw resume text.
            skill_weight (float, optional): Weight for keyword overlap (0-1). Defaults to 0.6.

        Returns:
            float: NLP score (0-100).
        """
        resume_clean = clean_text(resume_text)
        resume_embedding = self.model.encode([resume_clean], convert_to_numpy=True)[0]

        # --- Semantic similarity ---
        sim = cosine_similarity([resume_embedding], [self.jd_embedding])[0][0]

        # --- Keyword overlap ---
        resume_words = set(resume_clean.split())
        jd_words = set(self.jd_text.split())
        resume_skills = {w for w in resume_words if len(w) > 3}
        jd_skills = {w for w in jd_words if len(w) > 3}
        overlap = len(resume_skills & jd_skills) / len(jd_skills) if jd_skills else 0.0

        combined = (1 - skill_weight) * sim + skill_weight * overlap
        print(f"Combined score: {combined}")
        return float(np.clip(combined * 100, 0, 100))

    def llm_score(self, resume_text: str) -> float:
        """
        Compute a semantic search score using Groq LLM for a resume against a job description.

        The score is computed by splitting the resume into chunks of up to 500 words,
        generating a prompt for each chunk using the resume matching prompt template,
        and then invoking the LLM to score the prompt.

        The final score is the average of all chunk scores.

        Args:
            resume_text (str): Raw resume text.

        Returns:
            float: Semantic search score (0-100).
        """
        if not self.llm:
            return 0.0

        # --- Preprocess text ---
        max_chunk_words = 500  # safe chunk size
        resume_words = resume_text.split()
        num_chunks = math.ceil(len(resume_words) / max_chunk_words)

        chunk_scores = []

        for i in range(num_chunks):
            chunk_text = " ".join(
                resume_words[i * max_chunk_words : (i + 1) * max_chunk_words]
            )

            prompt_template = load_prompt("resume_matching_prompt")
            prompt = prompt_template.format(jd_text=self.jd_text, chunk_text=chunk_text)

            try:
                response = self.llm.invoke(prompt)
                import json

                result = json.loads(response.content.strip())
                chunk_scores.append(float(result.get("score", 0.0)))
            except Exception as e:
                print("LLM scoring failed for chunk: %s", e)
                chunk_scores.append(0.0)

        # --- Average all chunk scores ---
        final_score = sum(chunk_scores) / len(chunk_scores) if chunk_scores else 0.0
        return round(final_score, 2)
