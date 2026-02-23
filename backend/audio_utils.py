"""Audio utility functions for preprocessing and validation."""

import io
import os
import tempfile
from typing import Tuple

import numpy as np

SUPPORTED_EXTENSIONS = {".wav", ".mp3"}
MAX_DURATION_SECONDS = 30
MIN_DURATION_SECONDS = 1
TARGET_SAMPLE_RATE = 16000


def validate_audio_file(filename: str) -> bool:
    """Check if the uploaded file has a supported audio extension."""
    ext = os.path.splitext(filename)[1].lower()
    return ext in SUPPORTED_EXTENSIONS


def load_audio(file_bytes: bytes, filename: str) -> Tuple[np.ndarray, int]:
    """Load audio from bytes and return numpy array and sample rate.

    Resamples to TARGET_SAMPLE_RATE if needed and converts to mono.
    """
    import soundfile as sf

    buf = io.BytesIO(file_bytes)
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".mp3":
        import librosa

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            audio, sr = librosa.load(tmp_path, sr=TARGET_SAMPLE_RATE, mono=True)
        finally:
            os.unlink(tmp_path)
    else:
        audio, sr = sf.read(buf, dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != TARGET_SAMPLE_RATE:
            import librosa

            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
            sr = TARGET_SAMPLE_RATE

    return audio, sr


def validate_audio_duration(audio: np.ndarray, sr: int) -> bool:
    """Ensure audio duration is within acceptable bounds."""
    duration = len(audio) / sr
    return MIN_DURATION_SECONDS <= duration <= MAX_DURATION_SECONDS


def audio_to_wav_bytes(audio: np.ndarray, sr: int) -> bytes:
    """Convert numpy audio array to WAV bytes."""
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to [-1, 1] range."""
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio
