# 🎬 AI Video Assistant & Meeting Intelligence

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/Orchestration-LangChain%20LCEL-1C3C3C?logo=langchain&logoColor=white)](https://langchain.com)
[![OpenAI Whisper](https://img.shields.io/badge/STT-OpenAI%20Whisper-412991?logo=openai&logoColor=white)](https://github.com/openai/whisper)
[![ChromaDB](https://img.shields.io/badge/Vector%20Store-ChromaDB-orange)](https://trychroma.com)
[![Mistral AI](https://img.shields.io/badge/LLM-Mistral%20AI-FD5A24)](https://mistral.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end multimodal intelligence platform that converts lengthy YouTube videos, lecture recordings, and team meeting audio into structured, actionable intelligence. Features automated speech-to-text, Map-Reduce executive summarization, action item extraction, conversational RAG question-answering, and exportable PDF/Markdown reports.

---

## 📌 System Architecture

```mermaid
flowchart TD
    A[Input: YouTube URL / Audio File / Video File] --> B[Audio Preprocessor\n(yt-dlp + pydub 16kHz WAV)]
    B --> C[Audio Chunking Engine\n(10-min safety chunks)]
    
    C --> D{Language Router}
    D -- English --> E[Local OpenAI Whisper]
    D -- Hinglish / Indic --> F[Sarvam AI STT & Translation]
    
    E --> G[Unified Transcript]
    F --> G
    
    G --> H[LangChain Map-Reduce Summarizer]
    G --> I[Action Items & Decisions Extractor]
    G --> J[Recursive Character Splitter & MiniLM Embeddings]
    
    J --> K[(ChromaDB Vector Store)]
    K --> L[RAG Retrieval QA Chain\n(Mistral AI LCEL)]
    
    H --> M[Streamlit Dashboard & PDF/MD Exporter]
    I --> M
    L --> M
```

---

## 🚀 Key Engineering Highlights

- **Hybrid Speech-to-Text Pipeline**: Uses local **OpenAI Whisper** for zero-cost, high-fidelity offline English transcription, paired with a cloud fallback to **Sarvam AI** for Indic/Hinglish speech recognition and translation.
- **Map-Reduce Summarization Architecture**: Overcomes LLM context window bottlenecks by partitioning long transcripts, applying parallel chunk summarization, and aggregating insights through a synthesis prompt.
- **Production-Grade RAG Engine**: Implemented using **LangChain Expression Language (LCEL)**, embedding chunks with HuggingFace `all-MiniLM-L6-v2` into an isolated **ChromaDB** index to enable grounded, hallucination-resistant Q&A.
- **Resource & Memory Management**: Features automated disk cleanup for temporary `.wav` chunks and isolated per-session vector store states to prevent multi-meeting context contamination.
- **Multi-Format Export Engine**: Automatically generates standardized **Markdown** and styled **PDF** reports with executive summaries, tasks, and follow-ups.

---

## ✨ Features

- **Multimodal Input Support**: Paste any YouTube video link or upload local media files (`.mp4`, `.mp3`, `.wav`, `.m4a`, `.webm`, `.mkv`).
- **Automated Title & Executive Summary**: Concise meeting titles and bulleted high-level recaps.
- **Action Item & Owner Extraction**: Explicitly attributes tasks, assignees, and deadlines.
- **Key Decision Tracking**: Surfaces decisive choices made during the discussion.
- **Open Questions & Blockers**: Catalogs unresolved ambiguities for future syncs.
- **Interactive Chat with Your Meeting**: Talk directly to the recording transcript via a conversational RAG interface.
- **One-Click Export**: Export notes instantly to `.md` or `.pdf` format.

---

## 📂 Project Structure

```
ai_video_assistant/
├── app.py                  # Streamlit web dashboard with cyber-dark theme
├── main.py                 # CLI entry point for terminal-based processing
├── test.py                 # Health check, import verification & dry-run runner
├── requirements.txt        # Pinned Python package dependencies
├── .env.example            # Environment variables template
├── .gitignore              # Ignores temp audio, vector databases, and keys
├── core/
│   ├── config.py           # Centralized configuration & ChatMistralAI factory
│   ├── transcriber.py      # Whisper & Sarvam AI STT integration
│   ├── summarizer.py       # Map-Reduce executive summarization & title generation
│   ├── extractor.py        # Action items, decisions, and open question extractors
│   ├── vector_store.py     # ChromaDB vector store initialization & session isolation
│   └── rag_engine.py       # LCEL conversational RAG chain
└── utils/
    ├── audio_processor.py  # yt-dlp downloader, WAV converter, chunking & cleanup
    └── exporter.py         # Markdown and PDF report generation
```

---

## 🛠️ Getting Started

### 1. Prerequisites
- **Python 3.10+** (Python 3.10 – 3.12 recommended)
- **FFmpeg**: Required by `pydub` and `yt-dlp` for audio processing.
  - **Windows**: `winget install Gyan.FFmpeg` or download from [ffmpeg.org](https://ffmpeg.org/download.html)
  - **macOS**: `brew install ffmpeg`
  - **Linux**: `sudo apt install ffmpeg`

### 2. Clone and Setup Environment
```bash
# Clone the repository
git clone https://github.com/your-username/ai-video-assistant.git
cd ai-video-assistant

# Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Keys
Copy the `.env.example` file to `.env`:
```bash
cp .env.example .env
```
Edit `.env` and fill in your keys:
```env
# Required for Summaries, Extraction, and RAG Chat (Free tier available)
MISTRAL_API_KEY=your_mistral_api_key_here
MISTRAL_MODEL=mistral-small-latest

# Speech-to-Text configuration
WHISPER_MODEL=small

# Optional: Sarvam AI for Hinglish translation
SARVAM_API_KEY=your_sarvam_api_key_here
```

### 4. Verify Installation
Run the self-test health check script:
```bash
python test.py
```

---

## 🖥️ Running the Application

### Option A: Interactive Web UI (Streamlit)
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Option B: Command Line Interface (CLI)
```bash
python main.py
```

---

## 🤝 Contributing & License
Distributed under the MIT License. Contributions and feedback are welcome!
