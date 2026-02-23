"""TTS Engine using OpenAI TTS API.

This module provides text-to-speech generation via the OpenAI TTS-1-HD
model.  It keeps the local installation lightweight while supporting
cross-lingual English + Urdu synthesis.
"""

import logging
import os
import unicodedata
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


class TTSEngine:
    """Text-to-Speech engine backed by the OpenAI TTS API."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the TTS engine.

        Args:
            api_key: OpenAI API key.  Falls back to the
                     ``OPENAI_API_KEY`` environment variable.
        """
        # Priority: passed key > environment variable
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._client = None
        self._ready = False

    def load_model(self) -> bool:
        """Initialise the OpenAI client.

        Returns True when the API key is configured, False otherwise.
        """
        try:
            if not self.api_key:
                logger.error("OPENAI_API_KEY not found in environment.")
                return False
            self._client = OpenAI(api_key=self.api_key)
            self._ready = True
            logger.info("OpenAI TTS Engine initialised successfully.")
            return True
        except Exception as e:
            logger.error("Failed to initialize OpenAI Client: %s", e)
            return False

    def is_loaded(self) -> bool:
        """Check whether the OpenAI client is ready."""
        return self._ready

    def normalize_urdu_text(self, text: str) -> str:
        """Standardizes Urdu characters for optimal TTS synthesis."""
        text = unicodedata.normalize("NFC", text)
        replacements = {
            "\u06be": "\u0647",  # do-chashmi he
            "\u0649": "\u06cc",  # alef maksura
            "\u06d4": ". ",      # Urdu full stop
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text.strip()

    def generate_voice(self, text: str,
                       reference_audio_bytes: bytes = None,
                       language: str = "en") -> Optional[bytes]:
        """Generates audio using OpenAI TTS-1-HD model.

        Args:
            text: Text to synthesise (English or Urdu).
            reference_audio_bytes: Kept for API compatibility (not used
                by OpenAI).
            language: Language code ('en' or 'ur').

        Returns:
            Audio bytes, or None on failure.
        """
        if not self._ready or self._client is None:
            logger.error("Engine called before being loaded.")
            return None

        if language == "ur":
            text = self.normalize_urdu_text(text)

        try:
            logger.info("Requesting %s synthesis from OpenAI...", language)
            # 'onyx' is excellent for multilingual clarity
            response = self._client.audio.speech.create(
                model="tts-1-hd",
                voice="onyx",
                input=text,
            )
            return response.content
        except Exception as e:
            logger.error("OpenAI API error: %s", e)
            return None

    # Keep backward-compatible alias used by existing call-sites
    def generate_speech(self, text: str, language: str,
                        reference_audio_bytes: Optional[bytes] = None,
                        **_kwargs) -> Optional[bytes]:
        """Backward-compatible wrapper around ``generate_voice``."""
        return self.generate_voice(text, reference_audio_bytes, language)
