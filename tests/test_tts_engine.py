"""Tests for the TTS engine module."""

import numpy as np
import pytest

from backend.tts_engine import TTSEngine


class TestTTSEngine:
    def test_initialization(self):
        engine = TTSEngine(model_name="test-model", use_gpu=False,
                           quantize="none")
        assert engine.model_name == "test-model"
        assert engine.use_gpu is False
        assert engine.quantize == "none"
        assert engine.is_loaded() is False

    def test_normalize_urdu_text(self):
        engine = TTSEngine()
        # Test Urdu full stop replacement
        result = engine.normalize_urdu_text("سلام\u06d4")
        assert "\u06d4" not in result
        assert "." in result

    def test_normalize_urdu_character_mappings(self):
        engine = TTSEngine()
        # Test do-chashmi he normalization
        text_with_do_chashmi = "ی\u06beاں"
        result = engine.normalize_urdu_text(text_with_do_chashmi)
        assert "\u06be" not in result

    def test_is_loaded_default_false(self):
        engine = TTSEngine()
        assert engine.is_loaded() is False

    def test_extract_embedding_without_model(self):
        engine = TTSEngine()
        audio = np.random.randn(16000).astype(np.float32)
        result = engine.extract_speaker_embedding(audio, 16000)
        assert result is None

    def test_generate_without_model_or_fallback(self):
        """When neither model nor edge-tts is available, returns None."""
        engine = TTSEngine()
        # This will attempt edge-tts fallback which may or may not be installed
        # We just ensure no crash occurs
        result = engine.generate_speech("Hello", "en")
        # Result could be None or bytes depending on edge-tts availability
        assert result is None or isinstance(result, bytes)
