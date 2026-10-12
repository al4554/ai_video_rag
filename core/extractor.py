from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.config import get_llm


def build_extractor_chain(system_prompt: str, temperature: float = 0.2):
    """Build a standard LCEL information extraction chain."""
    llm = get_llm(temperature=temperature)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{text}"),
    ])
    return (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | prompt
        | llm
        | StrOutputParser()
    )


def extract_action_items(transcript: str) -> str:
    """Extract action items, assigning owners and deadlines when present."""
    chain = build_extractor_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all action items. For each item provide:\n"
        "- Task: description of the task\n"
        "- Owner: person responsible (or 'Unassigned')\n"
        "- Deadline: target date/time (or 'Not specified')\n\n"
        "Format as a numbered list. If none are found, say 'No action items found.'"
    )
    return chain.invoke(transcript)


def extract_key_decisions(transcript: str) -> str:
    """Extract all definitive decisions made during the meeting."""
    chain = build_extractor_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all key decisions made. Format as a numbered list with concise context. "
        "If none are found, say 'No key decisions found.'"
    )
    return chain.invoke(transcript)


def extract_questions(transcript: str) -> str:
    """Extract unresolved questions or follow-up topics."""
    chain = build_extractor_chain(
        "From the meeting transcript, extract all unresolved questions, "
        "open doubts, or topics requiring future follow-up. Format as a numbered list. "
        "If none are found, say 'No open questions found.'"
    )
    return chain.invoke(transcript)