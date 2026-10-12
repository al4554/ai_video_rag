import os
import shutil
from langchain_chroma import Chroma
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from core.config import CHROMA_DIR, EMBEDDING_MODEL

COLLECTION_NAME = "meeting_transcript"


def get_embeddings():
    """Initialize HuggingFace embedding model for vector representation."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )


def reset_vector_store(persist_dir: str = CHROMA_DIR):
    """Safely reset persisted vector store to prevent multi-session context contamination."""
    if os.path.exists(persist_dir):
        try:
            shutil.rmtree(persist_dir)
        except Exception:
            pass


def build_vector_store(
    transcript: str,
    collection_name: str = COLLECTION_NAME,
    reset: bool = True,
) -> Chroma:
    """Split transcript, generate embeddings, and build Chroma vector index."""
    print("🧠 Building vector store...")

    if reset:
        reset_vector_store(CHROMA_DIR)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )
    chunks = splitter.split_text(transcript)

    docs = [
        Document(
            page_content=chunk,
            metadata={"chunk_index": i, "total_chunks": len(chunks)},
        )
        for i, chunk in enumerate(chunks)
    ]

    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=CHROMA_DIR,
    )

    print(f"✅ Vector store ready with {len(docs)} chunk(s).")
    return vector_store


def load_vector_store(collection_name: str = COLLECTION_NAME) -> Chroma:
    """Load an existing persisted Chroma vector store."""
    embeddings = get_embeddings()
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )


def get_retriever(vector_store: Chroma, k: int = 4):
    """Return similarity search retriever for the vector store."""
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
