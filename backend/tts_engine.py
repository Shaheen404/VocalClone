"""TTS Engine for voice cloning using Fish Speech.

This module provides the voice cloning and text-to-speech generation
functionality. It abstracts the model loading (with quantization support
for free-tier GPUs) and exposes a simple interface for generating speech.

Supported models:
- Fish Speech V1.5 (primary, cross-lingual English + Urdu)
- Fallback to edge-tts for environments without GPU
"""

import io
import logging
import os
import tempfile
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Urdu phonetic normalization mappings for common transliteration issues
URDU_CHAR_NORMALIZATIONS = {
    "\u06be": "\u0647",  # do-chashmi he -> he
    "\u0649": "\u06cc",  # alef maksura -> ye
}


class TTSEngine:
    """Text-to-Speech engine with voice cloning capability."""

    def __init__(self, model_name: str = "fishaudio/fish-speech-1.5",
                 use_gpu: bool = True, quantize: str = "4bit"):
        """Initialize the TTS engine.

        Args:
            model_name: HuggingFace model identifier.
            use_gpu: Whether to attempt GPU acceleration.
            quantize: Quantization mode ('4bit', '8bit', or 'none').
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.quantize = quantize
        self.model = None
        self.processor = None
        self._device = "cpu"
        self._model_loaded = False

    def load_model(self) -> bool:
        """Load the TTS model with optional quantization.

        Returns True if model loaded successfully, False otherwise.
        Uses 4-bit/8-bit quantization for free-tier T4 GPU compatibility.
        """
        try:
            import torch
            from transformers import AutoModel, AutoProcessor

            device = "cpu"
            if self.use_gpu and torch.cuda.is_available():
                device = "cuda"

            load_kwargs = {"trust_remote_code": True}

            if device == "cuda" and self.quantize != "none":
                from transformers import BitsAndBytesConfig

                if self.quantize == "4bit":
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                    )
                elif self.quantize == "8bit":
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True,
                    )
            else:
                load_kwargs["device_map"] = device

            self.processor = AutoProcessor.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                self.model_name, **load_kwargs
            )
            self._device = device
            self._model_loaded = True
            logger.info("Model %s loaded on %s with %s quantization",
                        self.model_name, device, self.quantize)
            return True

        except Exception as e:
            logger.warning("Failed to load model %s: %s. "
                           "Falling back to edge-tts.", self.model_name, e)
            self._model_loaded = False
            return False

    def is_loaded(self) -> bool:
        """Check if the TTS model is loaded."""
        return self._model_loaded

    def extract_speaker_embedding(self, reference_audio: np.ndarray,
                                  sr: int) -> Optional[np.ndarray]:
        """Extract speaker embedding from reference audio.

        Args:
            reference_audio: Numpy array of the reference voice sample.
            sr: Sample rate of the reference audio.

        Returns:
            Speaker embedding as numpy array, or None on failure.
        """
        if not self._model_loaded:
            logger.warning("Model not loaded; cannot extract embedding.")
            return None

        try:
            import torch

            audio_tensor = torch.FloatTensor(reference_audio).unsqueeze(0)
            if self._device == "cuda":
                audio_tensor = audio_tensor.cuda()

            with torch.no_grad():
                embedding = self.model.encode_speaker(audio_tensor, sr)
            return embedding.cpu().numpy() if hasattr(embedding, 'cpu') else embedding

        except Exception as e:
            logger.error("Speaker embedding extraction failed: %s", e)
            return None

    def normalize_urdu_text(self, text: str) -> str:
        """Normalize Urdu text for better phonetic accuracy.

        Applies character normalization and handles common
        transliteration issues to reduce robotic output.
        """
        for old, new in URDU_CHAR_NORMALIZATIONS.items():
            text = text.replace(old, new)
        # Ensure proper spacing around Urdu punctuation
        text = text.replace("\u06d4", ". ")  # Urdu full stop
        return text.strip()

    def generate_speech(self, text: str, language: str,
                        reference_audio: Optional[np.ndarray] = None,
                        sr: int = 16000) -> Optional[bytes]:
        """Generate speech audio from text using the cloned voice.

        Args:
            text: Text to synthesize (English or Urdu).
            language: Language code ('en' or 'ur').
            reference_audio: Reference voice sample for cloning.
            sr: Sample rate of reference audio.

        Returns:
            WAV audio bytes or None on failure.
        """
        if language == "ur":
            text = self.normalize_urdu_text(text)

        # Try model-based generation first
        if self._model_loaded and reference_audio is not None:
            return self._generate_with_model(text, language,
                                             reference_audio, sr)

        # Fallback to edge-tts (no cloning, but functional)
        return self._generate_with_edge_tts(text, language)

    def _generate_with_model(self, text: str, language: str,
                             reference_audio: np.ndarray,
                             sr: int) -> Optional[bytes]:
        """Generate speech using the loaded model."""
        try:
            import torch
            import soundfile as sf

            audio_tensor = torch.FloatTensor(reference_audio).unsqueeze(0)
            if self._device == "cuda":
                audio_tensor = audio_tensor.cuda()

            with torch.no_grad():
                output = self.model.generate(
                    text=text,
                    language=language,
                    speaker_audio=audio_tensor,
                    speaker_sr=sr,
                )

            if hasattr(output, 'cpu'):
                output_np = output.cpu().numpy().squeeze()
            else:
                output_np = np.array(output).squeeze()

            buf = io.BytesIO()
            sf.write(buf, output_np, sr, format="WAV", subtype="PCM_16")
            buf.seek(0)
            return buf.read()

        except Exception as e:
            logger.error("Model generation failed: %s", e)
            return self._generate_with_edge_tts(text, language)

    def _generate_with_edge_tts(self, text: str,
                                language: str) -> Optional[bytes]:
        """Fallback TTS using edge-tts (no voice cloning)."""
        try:
            import asyncio
            import edge_tts

            voice = "en-US-AriaNeural" if language == "en" else "ur-PK-AsadNeural"
            communicate = edge_tts.Communicate(text, voice)

            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp_path = tmp.name
            tmp.close()

            try:
                asyncio.run(communicate.save(tmp_path))
                with open(tmp_path, "rb") as f:
                    return f.read()
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        except ImportError:
            logger.error("edge-tts not installed. Install with: "
                         "pip install edge-tts")
            return None
        except Exception as e:
            logger.error("edge-tts generation failed: %s", e)
            return None
