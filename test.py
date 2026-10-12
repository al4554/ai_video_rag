"""
AI Video Assistant — Verification & Health Check Script
Runs environment checks, imports validation, and dry-run tests on the RAG and extraction chains.
"""

import os
import sys
from dotenv import load_dotenv

# Ensure UTF-8 stdout encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

load_dotenv()


def check_environment():
    """Verify system prerequisites and environment variables."""
    print("[*] 1. Checking Environment Variables...")
    mistral_key = os.getenv("MISTRAL_API_KEY")
    sarvam_key = os.getenv("SARVAM_API_KEY")

    if not mistral_key:
        print("    [!] MISTRAL_API_KEY is not set in .env. LLM & RAG features require this key.")
    else:
        print(f"    [+] MISTRAL_API_KEY detected: {mistral_key[:4]}...{mistral_key[-3:]}")

    if not sarvam_key:
        print("    [i] SARVAM_API_KEY is not set (optional, only needed for Hinglish translation).")
    else:
        print(f"    [+] SARVAM_API_KEY detected: {sarvam_key[:4]}...{sarvam_key[-3:]}")


def test_imports():
    """Verify that all core modules import without syntax or dependency errors."""
    print("\n[*] 2. Validating Core Module Imports...")
    try:
        from core.config import MISTRAL_MODEL, WHISPER_MODEL
        from core.summarizer import summarize, generate_title
        from core.extractor import extract_action_items, extract_key_decisions, extract_questions
        from core.vector_store import build_vector_store
        from core.rag_engine import build_rag_chain, ask_question
        from utils.audio_processor import process_input
        from utils.exporter import generate_markdown_report, generate_pdf_report
        print(f"    [+] All modules imported successfully (Whisper: {WHISPER_MODEL}, LLM: {MISTRAL_MODEL})")
        return True
    except Exception as e:
        print(f"    [-] Import failed: {e}")
        return False


def test_synthetic_pipeline():
    """Run a synthetic end-to-end test on sample meeting text (avoids downloading video)."""
    print("\n[*] 3. Testing Synthetic Pipeline (Mock Transcript)...")
    sample_transcript = (
        "Project Sync Meeting: Sarah introduced the new Q3 migration timeline. "
        "The backend team decided to migrate from PostgreSQL 14 to 16 by next Friday. "
        "Alex is assigned to prepare the staging database and benchmark query latencies before Wednesday. "
        "Rachel asked whether we need downtime for the index rebuilds, which remains undecided. "
        "We agreed to conduct a follow-up review on Thursday at 2 PM."
    )

    if not os.getenv("MISTRAL_API_KEY"):
        print("    [!] Skipping LLM invocation (MISTRAL_API_KEY missing in .env).")
        return

    try:
        from core.summarizer import generate_title, summarize
        from core.extractor import extract_action_items, extract_key_decisions, extract_questions
        from core.rag_engine import build_rag_chain, ask_question
        from utils.exporter import generate_markdown_report

        print("    [>] Testing Title Generation...")
        title = generate_title(sample_transcript)
        print(f"        Title: {title}")

        print("    [>] Testing Summarizer...")
        summary = summarize(sample_transcript)
        print(f"        Summary generated ({len(summary)} chars).")

        print("    [>] Testing Extraction...")
        actions = extract_action_items(sample_transcript)
        decisions = extract_key_decisions(sample_transcript)
        questions = extract_questions(sample_transcript)
        print("        Extracted actions, decisions, and questions.")

        print("    [>] Testing Vector Store & RAG Chain...")
        rag_chain = build_rag_chain(sample_transcript)
        answer = ask_question(rag_chain, "Who is responsible for the staging database?")
        print(f"        RAG Answer: {answer}")

        result_dict = {
            "title": title,
            "transcript": sample_transcript,
            "summary": summary,
            "action_items": actions,
            "key_decisions": decisions,
            "open_questions": questions,
        }
        report = generate_markdown_report(result_dict)
        assert len(report) > 100, "Markdown report generation failed"
        print("    [+] Testing Exporter: Markdown report generated.")

        print("\n[+] All synthetic pipeline tests passed successfully!")

    except Exception as e:
        print(f"    [-] Synthetic pipeline test failed: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("AI Video Assistant -- Health Check & Self-Test")
    print("=" * 60)
    check_environment()
    if test_imports():
        test_synthetic_pipeline()
    print("=" * 60)