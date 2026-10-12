import os
import time
import tempfile
import streamlit as st
from dotenv import load_dotenv

from utils.audio_processor import process_input, cleanup_chunks
from utils.exporter import generate_markdown_report, generate_pdf_report
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question
from core.config import DOWNLOAD_DIR

load_dotenv()

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Video Assistant | Meeting Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

:root {
    --bg: #0a0a0f;
    --surface: #111118;
    --surface-2: #1a1a25;
    --border: #2a2a3a;
    --accent: #7c3aed;
    --accent-glow: #9f67ff;
    --accent-2: #06b6d4;
    --text: #e8e8f0;
    --text-muted: #7070a0;
    --success: #10b981;
    --warning: #f59e0b;
    --danger: #ef4444;
}

html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.stApp {
    background: var(--bg) !important;
}

.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image:
        linear-gradient(rgba(124, 58, 237, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(124, 58, 237, 0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Syne', sans-serif !important;
    color: var(--text) !important;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2rem, 5vw, 3.2rem);
    font-weight: 800;
    line-height: 1.1;
    margin: 0;
    background: linear-gradient(135deg, #ffffff 0%, var(--accent-glow) 50%, var(--accent-2) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: var(--text-muted);
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 0.5rem;
}

.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}

.card:hover {
    border-color: var(--accent);
}

.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, var(--accent), var(--accent-2));
}

.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.card-content {
    font-size: 0.875rem;
    line-height: 1.7;
    color: var(--text);
    white-space: pre-wrap;
}

.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.badge-purple { background: rgba(124,58,237,0.2); color: var(--accent-glow); border: 1px solid rgba(124,58,237,0.3); }
.badge-cyan   { background: rgba(6,182,212,0.15); color: var(--accent-2);    border: 1px solid rgba(6,182,212,0.3); }
.badge-green  { background: rgba(16,185,129,0.15); color: var(--success);    border: 1px solid rgba(16,185,129,0.3); }

.stTextInput > div > div > input,
.stSelectbox > div > div {
    background: var(--surface-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent), #5b21b6) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(124,58,237,0.4) !important;
}

.status-bar {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.65rem 0.9rem;
    background: var(--surface-2);
    border-radius: 8px;
    margin: 0.35rem 0;
    border: 1px solid var(--border);
    font-size: 0.8rem;
}

.status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}

.dot-active   { background: var(--accent-glow); box-shadow: 0 0 8px var(--accent-glow); animation: pulse 1.5s infinite; }
.dot-done     { background: var(--success); }
.dot-pending  { background: var(--border); }

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50%       { opacity: 0.4; }
}

.transcript-box {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.25rem;
    font-size: 0.82rem;
    line-height: 1.8;
    max-height: 350px;
    overflow-y: auto;
    color: var(--text-muted);
    white-space: pre-wrap;
    word-break: break-word;
}

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }
</style>
""", unsafe_allow_html=True)

# ─── Session State Init ──────────────────────────────────────────────────────────
for key, default in {
    "result": None,
    "chat_history": [],
    "pipeline_done": False,
    "pipeline_steps": {},
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def step_status(steps: dict, key: str) -> str:
    s = steps.get(key, "pending")
    if s == "active":
        return "dot-active"
    if s == "done":
        return "dot-done"
    return "dot-pending"


def render_step_bar(label: str, key: str, icon: str):
    css = step_status(st.session_state.pipeline_steps, key)
    st.markdown(f"""
    <div class="status-bar">
        <div class="status-dot {css}"></div>
        <span>{icon} {label}</span>
    </div>""", unsafe_allow_html=True)


# ─── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:1.6rem">🎬 AI Video</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Meeting Intelligence</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<span class="badge badge-purple">Input Source</span>', unsafe_allow_html=True)
    input_mode = st.radio("Choose Input Type", ["YouTube URL", "Upload Audio/Video", "Local File Path"], label_visibility="collapsed")

    source_path = ""
    if input_mode == "YouTube URL":
        source_path = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=...")
    elif input_mode == "Upload Audio/Video":
        uploaded_file = st.file_uploader("Upload media file", type=["mp4", "mp3", "wav", "m4a", "webm", "mkv"])
        if uploaded_file is not None:
            # Save uploaded file to temp directory
            temp_path = os.path.join(DOWNLOAD_DIR, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            source_path = temp_path
    else:
        source_path = st.text_input("Local File Path", placeholder="C:/path/to/meeting.mp4")

    language = st.selectbox(
        "Spoken Language",
        ["english", "hinglish"],
        index=0,
        help="'english' uses Whisper STT. 'hinglish' uses Sarvam AI with automatic English translation.",
    )

    run_btn = st.button("⚡ Analyse Meeting", use_container_width=True)

    if st.session_state.result:
        st.markdown("---")
        if st.button("🔄 Reset / New Video", use_container_width=True):
            st.session_state.result = None
            st.session_state.chat_history = []
            st.session_state.pipeline_done = False
            st.session_state.pipeline_steps = {}
            st.rerun()

    if st.session_state.pipeline_done:
        st.markdown("---")
        st.markdown('<span class="badge badge-green">Pipeline Status</span>', unsafe_allow_html=True)
        for step, icon, label in [
            ("audio",      "🔊", "Audio Processing"),
            ("transcript", "📝", "Transcription"),
            ("title",      "🏷️", "Title Generation"),
            ("summary",    "📋", "Summarisation"),
            ("extract",    "🔍", "Action & Decision Extraction"),
            ("rag",        "🧠", "RAG Vector Store"),
        ]:
            render_step_bar(label, step, icon)

# ─── Main Content Header ────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">AI Video Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Transcribe · Executive Summaries · Action Items · Interactive RAG Chat</div>', unsafe_allow_html=True)
st.markdown("---")

# ── Pipeline Execution ──────────────────────────────────────────────────────────
if run_btn:
    if not source_path.strip():
        st.error("Please provide a valid YouTube URL or select an audio/video file.")
    else:
        st.session_state.pipeline_done = False
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.pipeline_steps = {}

        progress_placeholder = st.empty()

        def update_step(key, state):
            st.session_state.pipeline_steps[key] = state

        chunks = []
        try:
            with progress_placeholder.container():
                st.info("⚙️ Meeting intelligence pipeline running — monitoring progress...")

            update_step("audio", "active")
            chunks = process_input(source_path)
            update_step("audio", "done")

            update_step("transcript", "active")
            transcript = transcribe_all(chunks, language=language)
            update_step("transcript", "done")

            update_step("title", "active")
            title = generate_title(transcript)
            update_step("title", "done")

            update_step("summary", "active")
            summary = summarize(transcript)
            update_step("summary", "done")

            update_step("extract", "active")
            action_items = extract_action_items(transcript)
            decisions = extract_key_decisions(transcript)
            questions = extract_questions(transcript)
            update_step("extract", "done")

            update_step("rag", "active")
            rag_chain = build_rag_chain(transcript)
            update_step("rag", "done")

            st.session_state.result = {
                "title": title,
                "transcript": transcript,
                "summary": summary,
                "action_items": action_items,
                "key_decisions": decisions,
                "open_questions": questions,
                "rag_chain": rag_chain,
            }
            st.session_state.pipeline_done = True
            progress_placeholder.success("✅ Analysis completed successfully!")
            time.sleep(0.5)
            progress_placeholder.empty()
            st.rerun()

        except Exception as e:
            for k in ["audio", "transcript", "title", "summary", "extract", "rag"]:
                if st.session_state.pipeline_steps.get(k) == "active":
                    st.session_state.pipeline_steps[k] = "pending"
            progress_placeholder.error(f"❌ Pipeline Execution Error: {str(e)}")
        finally:
            if chunks:
                cleanup_chunks(chunks)

# ── Results & Dashboard ─────────────────────────────────────────────────────────
if st.session_state.result:
    r = st.session_state.result

    # Header Card with Title & Quick Exports
    title_col, export_col1, export_col2 = st.columns([3, 1, 1], gap="small")
    with title_col:
        st.markdown(f"""
        <div class="card" style="padding:1rem 1.5rem">
            <div class="card-title">📌 Session Title</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.35rem;font-weight:700;color:var(--text)">
                {r['title']}
            </div>
        </div>""", unsafe_allow_html=True)

    with export_col1:
        md_data = generate_markdown_report(r)
        st.download_button(
            label="📥 Export .MD",
            data=md_data,
            file_name=f"{r['title'][:30].replace(' ', '_')}_notes.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with export_col2:
        pdf_data = generate_pdf_report(r)
        st.download_button(
            label="📑 Export PDF",
            data=pdf_data,
            file_name=f"{r['title'][:30].replace(' ', '_')}_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

    # Summary & Transcript
    col1, col2 = st.columns([3, 2], gap="medium")

    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">📋 Executive Summary</div>
            <div class="card-content">{r['summary']}</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        with st.expander("📝 Full Transcript", expanded=False):
            st.markdown(f'<div class="transcript-box">{r["transcript"]}</div>', unsafe_allow_html=True)

    # 3-column breakdown: Action items | Decisions | Questions
    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">✅ Action Items</div>
            <div class="card-content">{r['action_items']}</div>
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">🔑 Key Decisions</div>
            <div class="card-content">{r['key_decisions']}</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">❓ Open Questions</div>
            <div class="card-content">{r['open_questions']}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Interactive RAG Chat ──────────────────────────────────────────────────
    st.markdown('<div style="font-family:\'Syne\',sans-serif;font-size:1.3rem;font-weight:700;margin-bottom:0.75rem">💬 Chat with your Meeting</div>', unsafe_allow_html=True)
    st.caption("Ask questions about discussions, timeline, decisions, or speaker mentions using Retrieval-Augmented Generation (ChromaDB + Mistral AI).")

    # Display chat messages
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask any question regarding this meeting (e.g. 'What deadline was agreed upon?')..."):
        # Display user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching transcript context..."):
                answer = ask_question(r["rag_chain"], prompt)
                st.markdown(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})

else:
    # Empty State Display
    st.markdown("""
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:4rem 2rem;text-align:center">
        <div style="font-size:3.5rem;margin-bottom:1rem">🎬</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:700;color:var(--text);margin-bottom:0.5rem">
            Ready to Analyse Meeting
        </div>
        <div style="color:var(--text-muted);font-size:0.85rem;max-width:420px;line-height:1.7">
            Select a YouTube URL or upload a video/audio file in the sidebar, choose your spoken language, and click <strong>Analyse Meeting</strong>.
        </div>
        <div style="margin-top:2rem;display:flex;gap:1rem;flex-wrap:wrap;justify-content:center">
            <span class="badge badge-purple">Whisper & Sarvam STT</span>
            <span class="badge badge-cyan">Map-Reduce Summaries</span>
            <span class="badge badge-green">ChromaDB RAG QA</span>
        </div>
    </div>""", unsafe_allow_html=True)