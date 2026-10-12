import os
import glob
import yt_dlp
from pydub import AudioSegment
from core.config import DOWNLOAD_DIR

os.makedirs(DOWNLOAD_DIR, exist_ok=True)


def download_youtube_audio(url: str) -> str:
    """Download audio from YouTube and convert to 16kHz WAV format."""
    output_template = os.path.join(DOWNLOAD_DIR, "%(id)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = info.get("id", "audio")
        expected_wav = os.path.join(DOWNLOAD_DIR, f"{video_id}.wav")

        if os.path.exists(expected_wav):
            return expected_wav

        # Fallback search if ydl modified filename
        matched = glob.glob(os.path.join(DOWNLOAD_DIR, f"{video_id}*.wav"))
        if matched:
            return matched[0]

        prepared = ydl.prepare_filename(info)
        base = os.path.splitext(prepared)[0]
        return f"{base}.wav"


def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to 16kHz mono WAV format for Whisper."""
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(DOWNLOAD_DIR, f"{base_name}_16k.wav")

    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    audio.export(output_path, format="wav")
    return output_path


def chunk_audio(wav_path: str, chunk_minutes: int = 10) -> list:
    """Split WAV audio into smaller chunks (in minutes) for memory-efficient transcription."""
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000

    chunks = []
    base_name = os.path.splitext(wav_path)[0]

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        chunk_path = f"{base_name}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)

    return chunks


def process_input(source: str, chunk_minutes: int = 10) -> list:
    """Process either a YouTube URL or a local file path into transcription-ready WAV chunks."""
    if source.startswith("http://") or source.startswith("https://"):
        print("🔗 Detected YouTube URL. Downloading audio...")
        wav_path = download_youtube_audio(source)
    else:
        if not os.path.exists(source):
            raise FileNotFoundError(f"Input file not found: {source}")
        print("📁 Detected local media file. Converting to 16kHz mono WAV...")
        wav_path = convert_to_wav(source)

    print("✂️ Chunking audio...")
    chunks = chunk_audio(wav_path, chunk_minutes=chunk_minutes)
    print(f"✅ Audio ready — {len(chunks)} chunk(s) created.")
    return chunks


def cleanup_chunks(chunks: list, keep_original: bool = False):
    """Safely remove temporary audio chunks from disk."""
    for chunk in chunks:
        if os.path.exists(chunk):
            try:
                os.remove(chunk)
            except OSError:
                pass
