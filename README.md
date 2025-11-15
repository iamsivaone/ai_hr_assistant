
---

````markdown
# AI-Powered HR Assistant

## Project Overview
The **AI-Powered HR Assistant** is a Streamlit-based web application designed to help HR teams automatically evaluate and rank job candidates based on their resumes. The system supports both **NLP-based scoring** and **LLM-based scoring**, provides **resume summarization**, and includes a **chatbot interface** for querying candidate resumes. 

Key capabilities:  
- Upload and manage multiple resumes (PDF/DOCX/TXT).  
- Run candidate-job matching using NLP or LLM methods.  
- View ranked candidate scores.  
- Generate structured resume summaries.  
- Ask questions to a candidate-specific AI chatbot.

---

## Features
- **Resume Upload:** Supports PDF, DOCX, and TXT files.  
- **Job Description Input:** Paste or upload JD files.  
- **Matching Methods:**  
  - **NLP Matching:** Uses semantic embeddings and keyword overlap.  
  - **LLM Matching:** Uses a large language model (Groq OSS GPT-20B) for scoring.  
- **Ranked Results:** Sorted candidate scores with summary and chat options.  
- **Resume Summarization:** Extracts structured fields such as name, skills, experience, education, and recent work experience.  
- **Interactive Chat:** Ask AI questions about each candidate's resume.

---

## Setup Instructions

### Prerequisites
- Python 3.12
- Git (optional, if cloning the repository)
- LLM 

### Clone and Setup
```bash
# Clone repository
git clone <repository_url>
cd ai_hr_assistant

# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
````

---

## How to Run the Application

### Streamlit App

```bash
streamlit run app/streamlit_app.py
```

* The app will launch in your default browser.
* Sidebar provides instructions for uploading resumes and job descriptions.
* Use the **Matching Method** toggle to select NLP or LLM scoring before running matching.
* Click **Run Matching** to see ranked candidates.
* Click **Summary** to view a collapsible resume summary.
* Click **Chat** to interact with a candidate-specific AI assistant.

---

## API Keys / Credentials

This app may require API credentials for the LLM-based scoring and summarization features:

* **Groq OSS GPT-20B:** Make sure your environment has access to Groq's model using the appropriate API key or authentication method.
* Set environment variables if needed, for example:

  Use env_example.txt file


---

## Assumptions & Limitations

* Resumes should be in **PDF, DOCX, or TXT** formats.
* LLM-based scoring works best for resumes shorter than ~4–5 pages; longer resumes may require chunking.
* Keyword-based NLP scoring ignores words shorter than 3 characters.
* Summarization output may vary depending on resume formatting and content.
* Chatbot memory is session-based and does not persist after the session ends.

---

