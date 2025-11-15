from pathlib import Path
import uuid
import logging
import sys

from app.utils.utils import extract_text

sys.path.append(str(Path(__file__).resolve().parents[1]))
from typing import Dict, Any

import streamlit as st


from app.services.matcher import ResumeMatcher
from app.config.settings import settings
from app.services.summarizer import ResumeSummarizer
from app.services.chatbot import ResumeChatBot
from app.utils.file_utils import save_upload, is_allowed

# Configure basic logging
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

# Initialize session state containers
if "resumes" not in st.session_state:
    st.session_state["resumes"] = {}  # id -> resume dict

if "last_ranked" not in st.session_state:
    st.session_state["last_ranked"] = []

if "active_resume" not in st.session_state:
    st.session_state["active_resume"] = None

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = {}  # rid -> [{"q":..., "a":...}]

if "bots" not in st.session_state:
    st.session_state["bots"] = {}  # rid -> ResumeChatBot instance

if "matching_done" not in st.session_state:
    st.session_state["matching_done"] = False


# Helper UI functions


def sidebar_instructions() -> None:
    """
    Renders the sidebar instructions for the AI HR Assistant.

    The instructions are written in Markdown format and include the following:
    - Upload resumes in PDF/DOCX/TXT.
    - Paste or upload a Job Description (JD).
    - Click **Run matching** to score and rank candidates.
    - Use **Summarize** to extract structured fields from a resume.
    - Use **Chat** to ask questions about a selected resume (conversational memory kept in session).
    """
    st.sidebar.title("AI HR Assistant")
    st.sidebar.markdown(
        """
- Upload resumes in PDF/DOCX/TXT.
- Paste or upload a Job Description (JD).
- Click **Run matching** to score and rank candidates.
- Use **Summarize** to extract structured fields from a resume.
- Use **Chat** to ask questions about a selected resume (conversational memory kept in session).
"""
    )


def upload_resumes_widget() -> None:
    """
    Renders a Streamlit widget for uploading candidate resumes.

    The widget allows for multiple file uploads and accepts PDF, DOCX, and TXT files.
    If a file is already uploaded, it is skipped.
    If the file is not one of the allowed types, a warning is shown and the file is skipped.
    The uploaded file is saved to disk and its text is extracted using `extract_text`.
    The resume information is stored in session state with a unique ID.
    """
    st.header("Upload Candidate Resumes")

    ALLOWED_EXTENSIONS = ["pdf", "docx", "txt"]

    uploaded_files = st.file_uploader(
        "Upload resumes (PDF / DOCX / TXT). You can upload multiple files.",
        accept_multiple_files=True,
        type=ALLOWED_EXTENSIONS,
    )

    if uploaded_files:
        # Take only the last uploaded file
        uf = uploaded_files[-1]

        try:
            doc_name = Path(uf.name).stem

            # Skip if already uploaded
            already_uploaded = any(
                r.get("doc_name") == doc_name
                for r in st.session_state.get("resumes", {}).values()
            )
            if already_uploaded:
                # st.info(f"File '{uf.name}' already uploaded, skipping.")
                return

            # Validate file using util
            if not is_allowed(uf.name):
                st.warning(f"Skipped unsupported file type: {uf.name}")
                return

            # Save file to disk
            saved_path = save_upload(uf, settings.upload_dir)

            # Extract text from resume
            text = extract_text(saved_path)

            # Store resume info in session state
            cid = str(uuid.uuid4())
            st.session_state.setdefault("resumes", {})[cid] = {
                "id": cid,
                "doc_name": doc_name,
                "path": saved_path,
                "text": text,
            }

            # st.success(f"Uploaded: {uf.name}")

        except Exception as exc:
            logger.exception("Failed to save uploaded resume")
            st.error(f"Failed to save {getattr(uf, 'name', 'file')}: {exc}")


def jd_input_widget() -> str:
    """
    Renders a Streamlit widget for pasting a job description (JD) or uploading a JD file.

    The widget consists of a text area for pasting the JD and a file uploader for uploading a JD file.

    If a JD file is uploaded, the text is extracted from the file and populated into the text area.
    If the uploaded file is not a valid JD file, a warning is shown and the text area remains empty.

    The function returns the text from the text area.

    Returns:
        str: The text from the text area.
    """
    st.header("Job Description (JD)")
    jd_textarea = st.text_area("Paste full job description (required for matching)")
    uploaded_jd = st.file_uploader(
        "Or upload a JD (.txt)", type=["txt"], key="jd_upload"
    )

    if uploaded_jd is not None:
        try:
            jd_bytes = uploaded_jd.getvalue()
            jd_text_from_file = jd_bytes.decode("utf-8", errors="ignore")
            jd_textarea = jd_text_from_file
            # st.info("JD loaded from uploaded file.")
        except Exception:
            st.warning("Could not read uploaded JD file; please paste the JD manually.")

    return jd_textarea


def run_matching(jd_text: str) -> None:
    """
    Runs the matching process for the given job description against the uploaded resumes.

    Args:
        jd_text (str): The job description text.

    Returns:
        bool: True if matching is successful, False otherwise.

    Raises:
        Exception: If matching fails (e.g., due to invalid JD text or no resumes uploaded).
    """
    if not jd_text:
        st.warning("Please paste a job description to run matching.")
        return False

    resumes = list(st.session_state["resumes"].values())
    if not resumes:
        st.warning("No resumes uploaded yet. Upload resumes first.")
        return False

    try:
        with st.spinner("Running matching..."):
            matcher = ResumeMatcher(jd_text)
            ranked = []
            for resume in resumes:
                if st.session_state["match_mode"] == "LLM Matching":
                    score = matcher.llm_score(resume["text"])
                else:
                    score = matcher.nlp_score(resume["text"])
                ranked.append(
                    {
                        "id": resume["id"],
                        "doc_name": resume["doc_name"],
                        "text": resume["text"],
                        "path": resume["path"],
                        "score": round(score, 2),
                    }
                )

            # Sort candidates by score descending
            ranked.sort(key=lambda x: x["score"], reverse=True)
            st.session_state["last_ranked"] = ranked
            # st.success("✅ Matching complete.")
            return True
    except Exception as exc:
        logger.exception("Error during matching")
        st.error(f"Matching failed: {exc}")
        return False


def display_resume_summary(summary: dict, expanded: bool = True):
    """
    Displays a structured summary of a resume document.

    Args:
        summary (dict): A dictionary containing the structured summary of the resume text.
            The dictionary should contain the following keys:
                - name: The candidate's name.
                - professional_summary: A brief summary of the candidate's experience and skills.
                - total_experience: The total number of years of experience the candidate has.
                - key_skills: A comma-separated list of keywords representing the candidate's skills.
                - education: The candidate's highest level of education completed.
                - recent_work_experience: A list of the candidate's most recent work experiences, containing position, company, duration, and responsibilities.
        expanded (bool): Whether to display the summary in an expanded state. Defaults to True.

    Returns:
        None
    """
    if not summary:
        st.warning("No summary data available.")
        return

    rid = st.session_state.get("active_resume")
    key_expander = f"summary_expander_{rid}"
    if key_expander not in st.session_state:
        st.session_state[key_expander] = expanded

    # Expander with toggle
    with st.expander(
        f"Resume Summary: {summary.get('name', 'Unknown Candidate')}",
        expanded=st.session_state[key_expander],
    ):

        st.markdown(f"**Total Experience:** {summary.get('total_experience', 'N/A')}")
        st.markdown(f"**Education:** {summary.get('education', 'N/A')}")

        st.markdown("#### Professional Summary")
        st.write(summary.get("professional_summary", "No summary provided."))

        st.markdown("#### Key Skills")
        skills = summary.get("key_skills", [])
        if isinstance(skills, list) and skills:
            st.write(", ".join(skills))
        else:
            st.write("No skills listed.")

        st.markdown("#### Recent Work Experience")
        experiences = summary.get("recent_work_experience", [])
        if isinstance(experiences, list) and experiences:
            for exp in experiences:
                with st.expander(
                    f"{exp.get('position', 'Role')} at {exp.get('company', 'Company')}"
                ):
                    st.write(f"**Duration:** {exp.get('duration', 'N/A')}")
                    responsibilities = exp.get("responsibilities", [])
                    if responsibilities:
                        st.markdown("**Responsibilities:**")
                        for r in responsibilities:
                            st.markdown(f"- {r}")
        else:
            st.write("No recent work experience found.")


def show_ranked_results() -> None:
    """
    Displays a list of ranked candidates based on the last matching results.

    If no matching results are available, the function does nothing.

    The ranked list displays each candidate's document name and match score.

    For each candidate, two buttons are displayed: "Summary" and "Chat". Clicking
    "Summary" shows a structured summary of the candidate's resume, while
    clicking "Chat" opens a chat interface to ask questions about the candidate's
    experience and skills.

    The function uses Streamlit's session state to store the last matching results,
    the active resume ID, and whether to show the chat interface or summary.

    """
    if not st.session_state["last_ranked"]:
        return

    st.subheader("Ranked Candidates")

    for i, candidate in enumerate(st.session_state["last_ranked"]):
        cid = candidate["id"]
        name = candidate.get("doc_name", "Unnamed")
        score = candidate.get("score", 0.0)

        with st.container():
            cols = st.columns([4, 1, 1])
            with cols[0]:
                st.markdown(
                    f"""
                <div style="
                    border: 1px solid #ddd;
                    border-radius: 6px;
                    padding: 10px 16px;
                    margin-bottom: 8px;
                ">
                    Document Name: <b>{name}</b><br>
                    Match Score: <b>{score}%</b>
                </div>
                """,
                    unsafe_allow_html=True,
                )
            with cols[1]:
                if st.button("Summary", key=f"summarize_{cid}_{i}"):
                    summarizer = ResumeSummarizer()
                    with st.spinner("Summarizing..."):
                        summary = summarizer.summarize(candidate["text"])
                        st.session_state[f"summary_{cid}"] = summary
                        st.session_state["active_resume"] = cid
                        st.session_state["show_chat"] = False
                        st.session_state["show_summary"] = True
            with cols[2]:
                if st.button("Chat", key=f"chat_{cid}_{i}"):
                    st.session_state["active_resume"] = cid
                    st.session_state["show_chat"] = True
                    st.session_state["show_summary"] = False


def chat_widget() -> None:

    """
    Renders a Streamlit widget for chatting with a selected candidate's resume.

    The widget consists of a text input for asking a question, a send button, and a display area for the chat history.
    The chat history is persisted in the session state and is specific to each resume.

    The widget will display an error message if the chat query fails.

    :return: None
    """
    rid = st.session_state.get("active_resume")
    if not rid:
        st.info("Please select a resume to start chatting.")
        return

    resume = st.session_state["resumes"].get(rid)
    if not resume:
        st.error("Selected resume not found in session.")
        return

    st.header(f"Chat with {resume.get('doc_name', 'Candidate')} document")
    chat_col, info_col = st.columns([3, 1])

    # --- Initialize bot once per resume ---
    if rid not in st.session_state["bots"]:
        try:
            with st.spinner("Initializing bot..."):
                bot = ResumeChatBot(rid=rid, resume_text=resume["text"])
                st.session_state["bots"][rid] = bot
        except Exception as exc:
            logger.exception("Failed to initialize bot")
            st.error(f"Failed to initialize bot: {exc}")
            return

    bot = st.session_state["bots"][rid]

    # --- Initialize chat history ---
    if rid not in st.session_state["chat_history"]:
        st.session_state["chat_history"][rid] = []

    history = st.session_state["chat_history"][rid]

    # --- Chat input row (text input + send button side by side) ---
    chat_input_col, send_col = chat_col.columns([5, 1])

    with chat_input_col:
        question = st.text_input(
            "Ask a question about this candidate",
            key=f"q_input_{rid}",
            label_visibility="collapsed",
            placeholder="Type your question here...",
        )

    with send_col:
        send_clicked = st.button("Send", key=f"send_{rid}")

    if send_clicked and question.strip():
        try:
            with st.spinner("Fetching answer..."):
                answer = bot.query(question, chat_history=history)
                history.append({"q": question, "a": answer})
        except Exception as exc:
            logger.exception("Chat query failed")
            st.error(f"Failed to get an answer: {exc}")

    if history:
        # chat_col.markdown(f"**Q:** {history[-1]['q']}")
        # chat_col.markdown(f"**A:** {history[-1]['a']}")
        for msg in history:
            chat_col.markdown(f"**Q:** {msg['q']}")
            chat_col.markdown(f"**A:** {msg['a']}")


def main() -> None:
    """
    Main entry point for the AI-Powered HR Assistant Streamlit app.

    This function sets up the page configuration, renders the sidebar instructions,
    and defines the main layout of the app, including the widgets for uploading
    resumes, pasting a job description, running matching, showing ranked results,
    and displaying a candidate's summary or chat interface.
    """
    st.set_page_config(page_title="AI-Powered HR Assistant", layout="wide")
    sidebar_instructions()

    st.title("AI-Powered HR Assistant")

    left, right = st.columns([1.1, 1.2])

    with left:

        # Show Reset Matching only when matching is done
        if st.session_state["matching_done"]:
            if st.button("Reset Matching"):
                st.session_state["matching_done"] = False
                st.session_state["last_ranked"] = []
                st.session_state["active_resume"] = None
                st.session_state["show_chat"] = False
                st.session_state["show_summary"] = False
                st.session_state["resumes"] = {}  # <-- clear uploaded resumes
                st.session_state["chat_history"] = {}
                st.session_state["bots"] = {}
                st.rerun()  # ← force UI to refresh immediately

        # Show Upload + JD + Run Matching ONLY when matching not done
        if not st.session_state["matching_done"]:

            col1, col2 = st.columns(2)
            with col1:
                upload_resumes_widget()
            with col2:
                jd_text = jd_input_widget()

            st.markdown("<br>", unsafe_allow_html=True)

            # Toggle for matching mode
            st.subheader("Matching Method")
            match_mode = st.radio(
                "Choose matching type:",
                ["NLP Matching", "LLM Matching"],
                horizontal=True,
                key="match_mode",
            )

            if st.button("Run matching", key="run_matching_btn"):
                run_matching_res = run_matching(jd_text)
                if run_matching_res:
                    st.session_state["matching_done"] = True
                    st.rerun()  # <-- Force immediate refresh so widgets disappear

        # Show ranked results only when matching is done
        if st.session_state["matching_done"]:
            show_ranked_results()

    if st.session_state["last_ranked"]:
        with right:
            rid = st.session_state.get("active_resume")
            show_chat = st.session_state.get("show_chat", False)
            if rid:
                # Show summary if available
                summary_key = f"summary_{rid}"
                if summary_key in st.session_state:
                    display_resume_summary(st.session_state[summary_key])
                else:
                    st.info("Click **Summary** on a candidate to view details.")

                # Only show chat widget if Chat button was clicked
                if show_chat:
                    st.markdown("---")
                    chat_widget()
            else:
                st.info("Select a candidate to view summary or chat.")
    else:
        with right:
            st.info("Click **Run Matching** to view summary or chat.")

    # Footer
    st.markdown("---")


if __name__ == "__main__":
    main()
