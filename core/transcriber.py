import os
import requests
import whisper
from pydub import AudioSegment
from core.config import WHISPER_MODEL, SARVAM_MODEL, get_sarvam_api_key

# Sarvam's sync STT-translate API rejects audio longer than 30s.
# We slice each chunk into 25s pieces (with a 5s safety margin) before sending.
SARVAM_PIECE_SECONDS = 25
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"

_model = None


def load_model():
    """Lazily load the Whisper model into memory once."""
    global _model
    if _model is None:
        print(f"Loading Whisper model ({WHISPER_MODEL})...")
        _model = whisper.load_model(WHISPER_MODEL)
        print("Whisper model loaded successfully.")
    return _model


def transcribe_chunk_whisper(chunk_path: str) -> str:
    """Transcribe an audio chunk using local Whisper."""
    model = load_model()
    result = model.transcribe(chunk_path, task="transcribe")
    return result.get("text", "").strip()


def _send_to_sarvam(piece_path: str) -> str:
    """Send one <=30s WAV file to Sarvam API and return the English transcript."""
    api_key = get_sarvam_api_key()
    headers = {"api-subscription-key": api_key}

    with open(piece_path, "rb") as f:
        files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
        data = {"model": SARVAM_MODEL, "with_diarization": "false"}
        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )

    if not response.ok:
        print(f"\n❌ Sarvam API error: HTTP {response.status_code}")
        print(f"Response: {response.text}\n")
        response.raise_for_status()

    return response.json().get("transcript", "").strip()


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    """
    Split chunk into 25-second pieces for Sarvam's API limit, transcribe & translate each, and join results.
    """
    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000

    full_text = []
    total_pieces = (len(audio) + piece_ms - 1) // piece_ms

    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start : start + piece_ms]
        piece_path = f"{chunk_path}_sv_{i}.wav"
        piece.export(piece_path, format="wav")

        try:
            print(f"  → Sarvam piece {i + 1}/{total_pieces}...")
            text = _send_to_sarvam(piece_path)
            if text:
                full_text.append(text)
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return " ".join(full_text)


def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    """
    Route audio chunk to Whisper (local) or Sarvam (Indic/Hinglish).
    - english  → Whisper
    - hinglish → Sarvam (translates to English)
    """
    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)
    return transcribe_chunk_whisper(chunk_path)


def transcribe_all(chunks: list, language: str = "english") -> str:
    """Transcribe all provided audio chunks and return unified transcript."""
    full_transcript = []
    engine = "Sarvam AI (Hinglish -> English)" if language.lower() == "hinglish" else f"Whisper ({WHISPER_MODEL})"
    print(f"🎙️ Using {engine} for transcription.")

    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")
        text = transcribe_chunk(chunk, language=language)
        if text:
            full_transcript.append(text)

    print("✅ Transcription complete.")
    return " ".join(full_transcript).strip()
