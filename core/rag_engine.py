from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.config import get_llm
from core.vector_store import build_vector_store, load_vector_store, get_retriever

RAG_SYSTEM_PROMPT = """You are an expert meeting assistant. Answer the user's question 
based ONLY on the meeting transcript context provided below.

If the answer is not found in the context, say: 
"I could not find this information in the meeting transcript."

Always be concise, professional, and precise. If quoting someone, attribute it clearly.

Context from meeting transcript:
{context}"""


def format_docs(docs):
    """Concatenate retrieved document chunks."""
    return "\n\n".join([doc.page_content for doc in docs])


def _build_chain_from_retriever(retriever, temperature: float = 0.2):
    """Internal helper to construct the LCEL QA chain from a given retriever."""
    llm = get_llm(temperature=temperature)
    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    return (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )


def build_rag_chain(transcript: str, k: int = 4):
    """Build a vector index from the transcript and return an LCEL RAG QA chain."""
    vector_store = build_vector_store(transcript)
    retriever = get_retriever(vector_store, k=k)
    return _build_chain_from_retriever(retriever)


def load_rag_chain(k: int = 4):
    """Load the existing persisted vector index and return an LCEL RAG QA chain."""
    vector_store = load_vector_store()
    retriever = get_retriever(vector_store, k=k)
    return _build_chain_from_retriever(retriever)


def ask_question(rag_chain, question: str) -> str:
    """Invoke the RAG chain with a user query and return the answer."""
    question = question.strip()
    if not question:
        return "Please ask a specific question."
    return rag_chain.invoke(question)
