"""Audio utility functions for preprocessing and validation."""

import io
import os
import subprocess
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
    Uses ffmpeg for MP3 conversion (must be on the system PATH).
    """
    import soundfile as sf

    ext = os.path.splitext(filename)[1].lower()

    if ext == ".mp3":
        # Convert MP3 to WAV via ffmpeg
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_in:
            tmp_in.write(file_bytes)
            tmp_in_path = tmp_in.name

        tmp_out_path = tmp_in_path.replace(".mp3", ".wav")
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", tmp_in_path,
                    "-ar", str(TARGET_SAMPLE_RATE),
                    "-ac", "1",
                    tmp_out_path,
                ],
                check=True,
                capture_output=True,
            )
            audio, sr = sf.read(tmp_out_path, dtype="float32")
        finally:
            for p in (tmp_in_path, tmp_out_path):
                if os.path.exists(p):
                    os.unlink(p)
    else:
        buf = io.BytesIO(file_bytes)
        audio, sr = sf.read(buf, dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != TARGET_SAMPLE_RATE:
            # Resample using linear interpolation (lightweight, no librosa)
            duration = len(audio) / sr
            target_len = int(duration * TARGET_SAMPLE_RATE)
            indices = np.linspace(0, len(audio) - 1, target_len)
            audio = np.interp(indices, np.arange(len(audio)), audio).astype(
                np.float32
            )
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
