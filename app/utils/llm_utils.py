from langchain.chat_models import init_chat_model
from app.config.settings import settings


def get_llm():
    """
    Initializes a LangChain LLM instance using environment settings.

    Returns the initialized LLM instance if successful, otherwise None.

    Environment variables used:
        - LLM_MODEL: The name of the LLM model to use.
        - LLM_PROVIDER: The provider of the LLM model.
        - LLM_TEMPERATURE: The temperature to use when generating text with the LLM.

    If an exception occurs during initialization, it is caught and a message is printed to the console.
    """
    try:
        llm = init_chat_model(
            model=settings.llm_model,
            model_provider=settings.llm_provider,
            temperature=settings.llm_temperature,
        )
        print(f"LLM initialized: {settings.llm_model}")
        return llm
    except Exception as e:
        print("Failed to initialize LLM: %s", e)
        return None
