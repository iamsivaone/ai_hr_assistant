from typing import Dict
from app.config.settings import settings
from langchain.chat_models import init_chat_model

from app.utils.llm_utils import get_llm
from app.utils.utils import load_prompt


class ResumeSummarizer:

    def __init__(self):
        """
        Initializes the ResumeSummarizer instance.

        Tries to initialize a Groq LLM model instance. If successful, sets the `model` attribute to the initialized instance. If not, sets `model` to None and prints an error message.

        Raises:
            Exception: If the Groq LLM model initialization fails.
        """
        try:
            # Initialize LLM
            self.model = get_llm()
            print("Groq OSS 20B model initialized successfully.")
        except Exception as e:
            print("Failed to initialize Groq LLM model: %s", e)
            self.model = None

    def summarize(self, text: str) -> Dict[str, str]:
        """
        Generates a structured summary of a resume text using the Groq LLM model.

        Args:
            text (str): Resume text to be summarized.

        Returns:
            Dict[str, str]: A dictionary containing the structured summary of the resume text.
                The dictionary should contain the following keys:
                    - name: The candidate's name.
                    - professional_summary: A brief summary of the candidate's experience and skills.
                    - total_experience: The total number of years of experience the candidate has.
                    - key_skills: A comma-separated list of keywords representing the candidate's skills.
                    - education: The candidate's highest level of education completed.
                    - recent_work_experience: A list of the candidate's most recent work experiences, containing position, company, duration, and responsibilities.

        Raises:
            Exception: If the Groq LLM model initialization fails or if the summary generation fails.
        """
        if not text:
            print("Empty resume text passed to summarizer")
            return {}

        if not self.model:
            print("LLM model not initialized, returning placeholder summary")
            return {
                "name": "Unknown",
                "professional_summary": "",
                "total_experience": "",
                "key_skills": "",
                "education": "",
                "recent_work_experience": "",
            }

        # Build prompt for structured extraction
        prompt_template = load_prompt("resume_summarizer_prompt")
        prompt = prompt_template.format(text=text)

        try:
            response = self.model.invoke(prompt)
            text_output = response.content

            import json

            summary = json.loads(text_output.strip())
            return summary
        except Exception as e:
            print("Failed to generate summary via Groq LLM: %s", e)
            # Fallback to placeholder
            return {
                "name": "Unknown",
                "professional_summary": "",
                "total_experience": "",
                "key_skills": "",
                "education": "",
                "recent_work_experience": "",
            }
