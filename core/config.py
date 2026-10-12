import os
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI

# Load .env variables automatically on import
load_dotenv()

# Model and API Configuration
MISTRAL_MODEL = os.getenv("MISTRAL_MODEL", "mistral-small-latest")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOWNLOAD_DIR = os.path.join(BASE_DIR, "downloads")
CHROMA_DIR = os.path.join(BASE_DIR, "vector_db")

os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def get_mistral_api_key() -> str:
    """Retrieve Mistral API key or raise a descriptive error."""
    key = os.getenv("MISTRAL_API_KEY")
    if not key:
        raise ValueError(
            "MISTRAL_API_KEY is not set. Please set it in your .env file or environment variables."
        )
    return key


def get_sarvam_api_key() -> str:
    """Retrieve Sarvam API key or raise a descriptive error."""
    key = os.getenv("SARVAM_API_KEY")
    if not key:
        raise ValueError(
            "SARVAM_API_KEY is not set. Required for Hinglish transcription."
        )
    return key


def get_llm(temperature: float = 0.2) -> ChatMistralAI:
    """Central factory for ChatMistralAI instance."""
    api_key = get_mistral_api_key()
    return ChatMistralAI(
        model=MISTRAL_MODEL,
        mistral_api_key=api_key,
        temperature=temperature,
    )
