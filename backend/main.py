"""FastAPI backend for VocalClone - AI Voice Cloning Application.

Provides REST API endpoints for:
- Uploading reference voice samples (WAV/MP3)
- Generating TTS output in English and Urdu using cloned voice
- Health check and status endpoints
"""

import logging
import os
import tempfile
import uuid
from contextlib import asynccontextmanager
from typing import Dict

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .audio_utils import (
    audio_to_wav_bytes,
    load_audio,
    normalize_audio,
    validate_audio_duration,
    validate_audio_file,
)
from .tts_engine import TTSEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory storage for uploaded voice samples
voice_samples: Dict[str, Dict] = {}

# TTS Engine (initialized at startup)
tts_engine = TTSEngine(
    model_name=os.getenv("TTS_MODEL", "fishaudio/fish-speech-1.5"),
    use_gpu=os.getenv("USE_GPU", "true").lower() == "true",
    quantize=os.getenv("QUANTIZE", "4bit"),
)

# Output directory for generated audio
OUTPUT_DIR = os.getenv("OUTPUT_DIR", tempfile.mkdtemp(prefix="vocalclone_"))
os.makedirs(OUTPUT_DIR, exist_ok=True)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load TTS model on application startup."""
    logger.info("Starting VocalClone API...")
    success = tts_engine.load_model()
    if not success:
        logger.warning(
            "TTS model not loaded. Will use edge-tts fallback. "
            "For voice cloning, ensure a GPU is available and the "
            "model can be downloaded."
        )
    yield


app = FastAPI(
    title="VocalClone API",
    description="AI Voice Cloning - Generate speech in English and Urdu",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": tts_engine.is_loaded(),
        "model_name": tts_engine.model_name,
    }


@app.post("/api/upload")
async def upload_voice_sample(file: UploadFile = File(...)):
    """Upload a reference voice sample for cloning.

    Accepts WAV or MP3 files between 1-30 seconds in duration.
    Returns a sample_id to reference in TTS generation requests.
    """
    if not file.filename or not validate_audio_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload a WAV or MP3 file.",
        )

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    if len(file_bytes) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum size is 10MB.",
        )

    try:
        audio, sr = load_audio(file_bytes, file.filename)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process audio file: {str(e)}",
        )

    if not validate_audio_duration(audio, sr):
        duration = len(audio) / sr
        raise HTTPException(
            status_code=400,
            detail=f"Audio duration ({duration:.1f}s) must be between 1-30 seconds.",
        )

    audio = normalize_audio(audio)
    sample_id = str(uuid.uuid4())
    voice_samples[sample_id] = {
        "audio": audio,
        "sr": sr,
        "filename": file.filename,
        "duration": len(audio) / sr,
    }

    logger.info("Voice sample uploaded: %s (%.1fs)",
                sample_id, len(audio) / sr)

    return {
        "sample_id": sample_id,
        "filename": file.filename,
        "duration": round(len(audio) / sr, 1),
        "message": "Voice sample uploaded successfully.",
    }


@app.post("/api/generate")
async def generate_speech(
    sample_id: str = Form(...),
    text: str = Form(...),
    language: str = Form("en"),
):
    """Generate speech using the cloned voice.

    Args:
        sample_id: ID of the uploaded voice sample.
        text: Text to synthesize.
        language: Language code ('en' for English, 'ur' for Urdu).
    """
    if language not in ("en", "ur"):
        raise HTTPException(
            status_code=400,
            detail="Unsupported language. Use 'en' or 'ur'.",
        )

    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    if len(text) > 5000:
        raise HTTPException(
            status_code=400,
            detail="Text too long. Maximum 5000 characters.",
        )

    if sample_id not in voice_samples:
        raise HTTPException(
            status_code=404,
            detail="Voice sample not found. Please upload a sample first.",
        )

    sample = voice_samples[sample_id]

    logger.info("Generating speech: lang=%s, text_len=%d, sample=%s",
                language, len(text), sample_id)

    audio_bytes = tts_engine.generate_speech(
        text=text.strip(),
        language=language,
        reference_audio=sample["audio"],
        sr=sample["sr"],
    )

    if audio_bytes is None:
        raise HTTPException(
            status_code=500,
            detail="Failed to generate speech. Please try again.",
        )

    output_filename = f"{uuid.uuid4()}.wav"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    with open(output_path, "wb") as f:
        f.write(audio_bytes)

    return FileResponse(
        path=output_path,
        media_type="audio/wav",
        filename=f"vocalclone_{language}_{output_filename}",
    )


@app.get("/api/samples")
async def list_samples():
    """List all uploaded voice samples."""
    return {
        "samples": [
            {
                "sample_id": sid,
                "filename": data["filename"],
                "duration": round(data["duration"], 1),
            }
            for sid, data in voice_samples.items()
        ]
    }


@app.delete("/api/samples/{sample_id}")
async def delete_sample(sample_id: str):
    """Delete an uploaded voice sample."""
    if sample_id not in voice_samples:
        raise HTTPException(status_code=404, detail="Sample not found.")
    del voice_samples[sample_id]
    return {"message": "Sample deleted successfully."}


# Serve frontend static files in production
frontend_dist = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.isdir(frontend_dist):
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="frontend")
