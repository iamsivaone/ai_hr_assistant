from pathlib import Path


from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from app.utils.llm_utils import get_llm
from app.utils.text_utils import clean_text
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END, MessagesState
from langchain_core.messages import HumanMessage, AIMessage
from app.config.settings import settings
from app.utils.utils import load_prompt


class ResumeChatBot:
    """
    Conversational AI chatbot for interacting with a single candidate's resume.

    This class:
    - Builds or loads a FAISS vector index of the resume for semantic search.
    - Uses an LLM to answer questions about the resume content.
    - Maintains session-specific chat history.

    Attributes:
        bot_key (str): Unique identifier for this resume/chat instance.
        resume_text (str): Cleaned text content of the resume.
        llm: Language model instance initialized via `get_llm()`.
        embedder: SentenceTransformerEmbeddings for vectorization.
        vectorstore: FAISS index object for semantic search.
    """

    def __init__(self, rid: str, resume_text: str):
        """
        Initialize the ResumeChatBot instance.

        Args:
            rid (str): Unique ID for the resume instance.
            resume_text (str): Raw resume text.

        Raises:
            Exception: If LLM or vector embedding initialization fails.
        """
        print(f"Initializing ResumeChatBot with rid {rid} and resume text {resume_text}")
        self.bot_key = rid
        self.resume_text = clean_text(resume_text)
        try:
            # Initialize LLM
            print("Initializing LLM")
            self.llm = get_llm()
            if not self.llm:
                raise Exception("Failed to initialize LLM")
            print("LLM initialized successfully")

            # Initialize Embeddings
            print("Initializing Embeddings")
            self.embedder = SentenceTransformerEmbeddings(
                model_name=settings.embedding_model
            )
            if not self.embedder:
                raise Exception("Failed to initialize Embeddings")
            print("Embeddings initialized successfully")

            # Load FAISS index if exists, else build
            print("Loading FAISS index")
            if not self.load_index():
                print("Loading failed, building index instead")
                self.build_index()
            print("FAISS index loaded or built successfully")

            print("ResumeChatBot initialized successfully.")
        except Exception as exc:
            print(f"Failed to initialize chatbot: {exc}")
            self.llm = None
            self.embedder = None

    def get_index_path(self) -> Path:
        """
        Get the path to the FAISS index for this resume.

        Returns:
            Path: Full path to the FAISS index file.
        """
        print(f"Getting index path for bot key {self.bot_key}")
        path = settings.index_dir / f"{self.bot_key}_faiss_index"
        print(f"Index path is {path}")
        return path

    def build_index(self):
        """
        Build the FAISS index for this resume using the given text.

        The index is constructed by recursively splitting the text into chunks
        of size up to `settings.max_chunk`, with an overlap of
        `settings.chunk_overlap` characters between successive chunks.

        The resulting chunks are then embedded using the `self.embedder`
        and indexed using FAISS.

        The index is saved to disk at the path specified by `self.get_index_path()`.
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.max_chunk,
            chunk_overlap=settings.chunk_overlap,
        )
        chunks = splitter.split_text(self.resume_text)

        self.vectorstore = FAISS.from_texts(chunks, embedding=self.embedder)

        index_path = self.get_index_path()
        index_path.parent.mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(str(index_path))

    def load_index(self) -> bool:
        """
        Load a pre-existing FAISS index from disk.

        If the index exists at the path specified by `self.get_index_path()`,
        it is loaded into memory and assigned to `self.vectorstore`. If not,
        `self.vectorstore` remains `None`.

        Returns:
            bool: Whether the index was successfully loaded.
        """
        index_path = self.get_index_path()
        print(f"Loading FAISS index from {index_path}")
        if index_path.exists():
            print(f"Found index at {index_path}")
            try:
                self.vectorstore = FAISS.load_local(
                    str(index_path), self.embedder, allow_dangerous_deserialization=True
                )
                print(f"Loaded FAISS index from {index_path}")
                return True
            except Exception as e:
                print(f"Failed to load FAISS index from {index_path}: {e}")
        print(f"No index found at {index_path}")
        return False

    def query(self, question: str, chat_history: list, top_k: int = 5) -> str:
        """
        Given a question, retrieve the top-k relevant chunks from the resume text
        and generate a response based on the chat history and context.

        Args:
            question (str): The question to generate a response for.
            chat_history (list): List of previous chat messages.
            top_k (int, optional): Number of relevant chunks to retrieve. Defaults to 5.

        Returns:
            str: The generated response based on the context and chat history.
        """
        print(f"Querying for question '{question}'")
        try:
            if not self.vectorstore:
                raise ValueError("FAISS index not built")

            # --- Retrieve top-k relevant chunks ---
            docs = self.vectorstore.similarity_search(query=question, k=top_k)
            print(f"Retrieved {len(docs)} relevant chunks for question '{question}'")
            context = "\n".join([d.page_content for d in docs])

            # Build prompt including context
            prompt_template_str = load_prompt("resume_chat_prompt")
            prompt_template = ChatPromptTemplate.from_template(prompt_template_str)

            # Format chat history as text
            history_text = ""
            for msg in chat_history:
                history_text += f"User: {msg['q']}\nAssistant: {msg['a']}\n"

            prompt = prompt_template.format_messages(
                context=context, history=history_text.strip(), question=question
            )
            print(f"Generated prompt for question '{question}': {prompt}")

            # Define LangGraph node
            def call_model(state: MessagesState):
                res = self.llm.invoke(prompt)
                print(f"LLM generated response for question '{question}': {res.content}")
                return {"messages": [AIMessage(content=res.content)]}

            workflow = StateGraph(state_schema=MessagesState)
            workflow.add_node("model", call_model)
            workflow.add_edge(START, "model")
            workflow.add_edge("model", END)

            # memory = InMemorySaver()
            # app = workflow.compile(checkpointer=memory)

            app = workflow.compile()

            config = {"configurable": {"thread_id": self.bot_key}}
            input = [HumanMessage(question)]
            output = app.invoke({"messages": input}, config)

            return output["messages"][-1].content
        except Exception as e:
            print(f"Failed to query for question '{question}': {e}")
            return ""
