"""TTS Engine for voice cloning using Fish Audio SDK.

This module provides voice cloning and text-to-speech generation via the
Fish Audio cloud API.  It keeps the local installation lightweight (no
PyTorch / CUDA wheels) while supporting cross-lingual English + Urdu
synthesis with reference-audio voice cloning.
"""

import logging
import os
import unicodedata
from typing import Optional

from fish_audio_sdk import ReferenceAudio, Session, TTSRequest

logger = logging.getLogger(__name__)

# Urdu phonetic normalization mappings for common transliteration issues
URDU_CHAR_NORMALIZATIONS = {
    "\u06be": "\u0647",  # do-chashmi he -> he
    "\u0649": "\u06cc",  # alef maksura -> ye
}


class TTSEngine:
    """Text-to-Speech engine backed by the Fish Audio SDK."""

    def __init__(self, api_key: Optional[str] = None,
                 base_url: str = "https://api.fish.audio"):
        """Initialize the TTS engine.

        Args:
            api_key: Fish Audio API key.  Falls back to the
                     ``FISH_AUDIO_API_KEY`` environment variable.
            base_url: Fish Audio API base URL.
        """
        self.api_key = api_key or os.getenv("FISH_AUDIO_API_KEY", "")
        self.base_url = base_url
        self._session: Optional[Session] = None
        self._ready = False

    def load_model(self) -> bool:
        """Initialise the Fish Audio SDK session.

        Returns True when the API key is configured, False otherwise.
        """
        if not self.api_key:
            logger.warning(
                "FISH_AUDIO_API_KEY not set. TTS generation will be "
                "unavailable until a valid API key is provided."
            )
            self._ready = False
            return False

        try:
            self._session = Session(
                apikey=self.api_key,
                base_url=self.base_url,
            )
            self._ready = True
            logger.info("Fish Audio SDK session initialised "
                        "(base_url=%s)", self.base_url)
            return True
        except Exception as e:
            logger.error("Failed to initialise Fish Audio session: %s", e)
            self._ready = False
            return False

    def is_loaded(self) -> bool:
        """Check whether the SDK session is ready."""
        return self._ready

    def normalize_urdu_text(self, text: str) -> str:
        """Normalize Urdu text for better phonetic accuracy.

        Applies Unicode NFC normalization (prevents character
        disconnection on Python 3.13), character-level mappings, and
        Urdu punctuation spacing.
        """
        text = unicodedata.normalize("NFC", text)
        for old, new in URDU_CHAR_NORMALIZATIONS.items():
            text = text.replace(old, new)
        # Ensure proper spacing around Urdu punctuation
        text = text.replace("\u06d4", ". ")  # Urdu full stop
        return text.strip()

    def generate_voice(self, text: str,
                       reference_audio_bytes: bytes,
                       language: str = "en",
                       output_format: str = "wav") -> Optional[bytes]:
        """Generate speech with voice cloning via the Fish Audio SDK.

        Args:
            text: Text to synthesise (English or Urdu).
            reference_audio_bytes: Raw bytes of the reference voice
                sample (WAV or MP3).
            language: Language code ('en' or 'ur').
            output_format: Output audio format ('wav' or 'mp3').

        Returns:
            Audio bytes in the requested format, or None on failure.
        """
        if language == "ur":
            text = self.normalize_urdu_text(text)

        if not self._ready or self._session is None:
            logger.error("Fish Audio session not initialised.")
            return None

        try:
            request = TTSRequest(
                text=text,
                format=output_format,
                references=[
                    ReferenceAudio(audio=reference_audio_bytes, text=""),
                ],
            )

            audio_chunks: list[bytes] = []
            for chunk in self._session.tts(request):
                audio_chunks.append(chunk)

            if not audio_chunks:
                logger.error("Fish Audio SDK returned no audio data.")
                return None

            return b"".join(audio_chunks)

        except Exception as e:
            logger.error("Fish Audio TTS generation failed: %s", e)
            return None

    # Keep backward-compatible alias used by existing call-sites
    def generate_speech(self, text: str, language: str,
                        reference_audio_bytes: Optional[bytes] = None,
                        **_kwargs) -> Optional[bytes]:
        """Backward-compatible wrapper around ``generate_voice``."""
        if reference_audio_bytes is None:
            logger.error("Reference audio is required for voice cloning.")
            return None
        return self.generate_voice(
            text=text,
            reference_audio_bytes=reference_audio_bytes,
            language=language,
        )
