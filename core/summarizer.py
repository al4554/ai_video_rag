from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.config import get_llm


def split_transcript(transcript: str) -> list:
    """Split long transcripts into chunks suitable for Map-Reduce summarization."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
    )
    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    """
    Generate an executive summary using a Map-Reduce style summarization pipeline.
    If the transcript is short, summarizes in a single pass to save tokens and latency.
    """
    llm = get_llm(temperature=0.3)
    chunks = split_transcript(transcript)

    if not chunks:
        return "No transcript content available to summarize."

    # Single-pass fast path for short transcripts
    if len(chunks) == 1:
        direct_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are an expert executive meeting summarizer. Provide a concise, "
                "well-structured summary of the meeting highlights using clear bullet points.",
            ),
            ("human", "{text}"),
        ])
        direct_chain = (
            RunnablePassthrough()
            | RunnableLambda(lambda x: {"text": x})
            | direct_prompt
            | llm
            | StrOutputParser()
        )
        return direct_chain.invoke(transcript)

    # Multi-pass Map-Reduce for long transcripts
    map_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize this portion of a meeting transcript concisely."),
        ("human", "{text}"),
    ])
    map_chain = map_prompt | llm | StrOutputParser()

    chunk_summaries = [map_chain.invoke({"text": chunk}) for chunk in chunks]
    combined = "\n\n".join(chunk_summaries)

    reduce_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert meeting summarizer. Combine these section summaries "
            "into one comprehensive executive meeting summary in structured bullet points.",
        ),
        ("human", "{text}"),
    ])

    reduce_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | reduce_prompt
        | llm
        | StrOutputParser()
    )

    return reduce_chain.invoke(combined)


def generate_title(transcript: str) -> str:
    """Generate a crisp, professional meeting title (max 8 words) from the transcript."""
    llm = get_llm(temperature=0.2)

    title_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages([
            (
                "system",
                "Based on the meeting transcript, generate a short professional meeting title "
                "(maximum 8 words). Return only the title without quotes or preamble.",
            ),
            ("human", "{text}"),
        ])
        | llm
        | StrOutputParser()
    )

    return title_chain.invoke(transcript[:2000]).strip()
