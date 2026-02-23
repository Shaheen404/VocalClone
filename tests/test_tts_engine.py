"""Tests for the TTS engine module."""

import pytest

from backend.tts_engine import TTSEngine


class TestTTSEngine:
    def test_initialization(self):
        engine = TTSEngine(api_key="test-key")
        assert engine.api_key == "test-key"
        assert engine.is_loaded() is False

    def test_initialization_no_key(self):
        engine = TTSEngine(api_key="")
        assert engine.api_key is None
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

    def test_normalize_urdu_nfc(self):
        engine = TTSEngine()
        # NFC normalization should be applied
        result = engine.normalize_urdu_text("ا")
        assert isinstance(result, str)

    def test_is_loaded_default_false(self):
        engine = TTSEngine()
        assert engine.is_loaded() is False

    def test_load_model_without_key(self):
        engine = TTSEngine(api_key="")
        result = engine.load_model()
        assert result is False
        assert engine.is_loaded() is False

    def test_generate_voice_without_session(self):
        engine = TTSEngine(api_key="")
        result = engine.generate_voice("Hello", b"audio-bytes")
        assert result is None

    def test_generate_speech_backward_compat(self):
        engine = TTSEngine(api_key="")
        result = engine.generate_speech("Hello", "en",
                                        reference_audio_bytes=b"audio")
        assert result is None

    def test_generate_speech_no_reference(self):
        engine = TTSEngine(api_key="")
        result = engine.generate_speech("Hello", "en")
        # With OpenAI, reference audio is optional; returns None because
        # the engine is not loaded (no API key).
        assert result is None
