import sys
from dotenv import load_dotenv

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


from utils.audio_processor import process_input, cleanup_chunks
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()


def run_pipeline(source: str, language: str = "english") -> dict:
    """Execute end-to-end meeting analysis pipeline via CLI."""
    print("\n🎬 Starting AI Video Assistant Pipeline...")

    chunks = []
    try:
        chunks = process_input(source)
        transcript = transcribe_all(chunks, language=language)

        if not transcript.strip():
            raise RuntimeError("Transcription produced empty result.")

        print(f"\n📝 Transcript preview ({min(len(transcript), 300)} chars):\n{transcript[:300]}...\n")

        print("🏷️ Generating meeting title...")
        title = generate_title(transcript)

        print("📋 Generating executive summary...")
        summary = summarize(transcript)

        print("🔍 Extracting action items, decisions, and questions...")
        action_items = extract_action_items(transcript)
        decisions = extract_key_decisions(transcript)
        questions = extract_questions(transcript)

        print("🧠 Building vector store for interactive RAG...")
        rag_chain = build_rag_chain(transcript)

        return {
            "title": title,
            "transcript": transcript,
            "summary": summary,
            "action_items": action_items,
            "key_decisions": decisions,
            "open_questions": questions,
            "rag_chain": rag_chain,
        }
    finally:
        if chunks:
            cleanup_chunks(chunks)


def main():
    print("=" * 60)
    print("🎬 AI Video Assistant & Meeting Intelligence (CLI)")
    print("=" * 60)

    source = input("Enter YouTube URL or local file path: ").strip()
    if not source:
        print("❌ Error: Source cannot be empty.")
        sys.exit(1)

    lang_choice = input("Spoken Language ([1] English / [2] Hinglish) [Default: 1]: ").strip()
    language = "hinglish" if lang_choice == "2" or lang_choice.lower() == "hinglish" else "english"

    try:
        result = run_pipeline(source, language)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(f"📌 Title: {result['title']}")
    print("=" * 60)
    print(f"\n📋 Executive Summary:\n{result['summary']}")
    print(f"\n✅ Action Items:\n{result['action_items']}")
    print(f"\n🔑 Key Decisions:\n{result['key_decisions']}")
    print(f"\n❓ Open Questions:\n{result['open_questions']}")
    print("=" * 60)

    # Interactive RAG Session
    print("\n💬 Chat with your meeting context (type 'exit' or 'q' to quit)\n")
    rag_chain = result["rag_chain"]
    while True:
        try:
            query = input("You: ").strip()
            if query.lower() in ["exit", "quit", "q"]:
                print("👋 Exiting interactive session.")
                break
            if not query:
                continue
            answer = ask_question(rag_chain, query)
            print(f"\n🤖 Assistant: {answer}\n")
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Session closed.")
            break


if __name__ == "__main__":
    main()