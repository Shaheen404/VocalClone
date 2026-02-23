"""Tests for the audio utilities module."""

import io
import struct

import numpy as np
import pytest

from backend.audio_utils import (
    audio_to_wav_bytes,
    normalize_audio,
    validate_audio_duration,
    validate_audio_file,
)


class TestValidateAudioFile:
    def test_wav_accepted(self):
        assert validate_audio_file("sample.wav") is True

    def test_mp3_accepted(self):
        assert validate_audio_file("sample.mp3") is True

    def test_uppercase_accepted(self):
        assert validate_audio_file("sample.WAV") is True

    def test_unsupported_rejected(self):
        assert validate_audio_file("sample.txt") is False

    def test_no_extension_rejected(self):
        assert validate_audio_file("sample") is False

    def test_ogg_rejected(self):
        assert validate_audio_file("sample.ogg") is False


class TestValidateAudioDuration:
    def test_valid_duration(self):
        sr = 16000
        audio = np.zeros(sr * 10)  # 10 seconds
        assert validate_audio_duration(audio, sr) is True

    def test_too_short(self):
        sr = 16000
        audio = np.zeros(int(sr * 0.5))  # 0.5 seconds
        assert validate_audio_duration(audio, sr) is False

    def test_too_long(self):
        sr = 16000
        audio = np.zeros(sr * 60)  # 60 seconds
        assert validate_audio_duration(audio, sr) is False

    def test_boundary_min(self):
        sr = 16000
        audio = np.zeros(sr * 1)  # exactly 1 second
        assert validate_audio_duration(audio, sr) is True

    def test_boundary_max(self):
        sr = 16000
        audio = np.zeros(sr * 30)  # exactly 30 seconds
        assert validate_audio_duration(audio, sr) is True


class TestNormalizeAudio:
    def test_normalizes_to_unit_range(self):
        audio = np.array([0.0, 0.5, -0.5, 0.25])
        result = normalize_audio(audio)
        assert np.max(np.abs(result)) == pytest.approx(1.0)

    def test_silent_audio(self):
        audio = np.zeros(100)
        result = normalize_audio(audio)
        assert np.all(result == 0)

    def test_already_normalized(self):
        audio = np.array([1.0, -1.0, 0.5])
        result = normalize_audio(audio)
        np.testing.assert_array_almost_equal(result, audio)


class TestAudioToWavBytes:
    def test_returns_bytes(self):
        audio = np.random.randn(16000).astype(np.float32)
        result = audio_to_wav_bytes(audio, 16000)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_wav_header(self):
        audio = np.random.randn(16000).astype(np.float32)
        result = audio_to_wav_bytes(audio, 16000)
        assert result[:4] == b"RIFF"
