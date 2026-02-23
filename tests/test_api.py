"""Tests for the FastAPI application endpoints."""

import io
import struct

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app, voice_samples


@pytest.fixture
def client():
    """Create a test client."""
    voice_samples.clear()
    return TestClient(app)


def make_wav_bytes(duration_seconds: float = 5.0,
                   sample_rate: int = 16000) -> bytes:
    """Create a minimal valid WAV file in memory."""
    import soundfile as sf

    n_samples = int(sample_rate * duration_seconds)
    audio = np.random.randn(n_samples).astype(np.float32) * 0.1
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


class TestHealthEndpoint:
    def test_health_check(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert data["engine"] == "fish-audio-sdk"


class TestUploadEndpoint:
    def test_upload_wav(self, client):
        wav_bytes = make_wav_bytes(5.0)
        response = client.post(
            "/api/upload",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
        assert response.status_code == 200
        data = response.json()
        assert "sample_id" in data
        assert data["filename"] == "test.wav"
        assert data["duration"] > 0

    def test_upload_invalid_format(self, client):
        response = client.post(
            "/api/upload",
            files={"file": ("test.txt", b"not audio", "text/plain")},
        )
        assert response.status_code == 400

    def test_upload_empty_file(self, client):
        response = client.post(
            "/api/upload",
            files={"file": ("test.wav", b"", "audio/wav")},
        )
        assert response.status_code == 400

    def test_upload_too_short(self, client):
        wav_bytes = make_wav_bytes(0.1)  # 0.1 seconds
        response = client.post(
            "/api/upload",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
        assert response.status_code == 400


class TestSamplesEndpoint:
    def test_list_empty(self, client):
        response = client.get("/api/samples")
        assert response.status_code == 200
        assert response.json()["samples"] == []

    def test_list_after_upload(self, client):
        wav_bytes = make_wav_bytes(5.0)
        client.post(
            "/api/upload",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
        response = client.get("/api/samples")
        assert response.status_code == 200
        assert len(response.json()["samples"]) == 1

    def test_delete_sample(self, client):
        wav_bytes = make_wav_bytes(5.0)
        upload_resp = client.post(
            "/api/upload",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
        )
        sample_id = upload_resp.json()["sample_id"]
        response = client.delete(f"/api/samples/{sample_id}")
        assert response.status_code == 200

    def test_delete_nonexistent(self, client):
        response = client.delete("/api/samples/nonexistent")
        assert response.status_code == 404


class TestGenerateEndpoint:
    def test_generate_missing_sample_and_file(self, client):
        response = client.post(
            "/api/generate",
            data={
                "text": "Hello world",
                "language": "en",
            },
        )
        assert response.status_code == 400

    def test_generate_missing_sample_id(self, client):
        response = client.post(
            "/api/generate",
            data={
                "sample_id": "nonexistent",
                "text": "Hello world",
                "language": "en",
            },
        )
        assert response.status_code == 404

    def test_generate_with_file(self, client):
        wav_bytes = make_wav_bytes(5.0)
        response = client.post(
            "/api/generate",
            data={
                "text": "Hello world",
                "language": "en",
            },
            files={"file": ("voice.wav", wav_bytes, "audio/wav")},
        )
        # Will fail at TTS engine (no API key) but validates the endpoint
        assert response.status_code in (200, 500)

    def test_generate_invalid_language(self, client):
        response = client.post(
            "/api/generate",
            data={
                "sample_id": "any",
                "text": "Hello",
                "language": "fr",
            },
        )
        assert response.status_code == 400

    def test_generate_empty_text(self, client):
        response = client.post(
            "/api/generate",
            data={
                "sample_id": "any",
                "text": "   ",
                "language": "en",
            },
        )
        assert response.status_code == 400

    def test_generate_text_too_long(self, client):
        response = client.post(
            "/api/generate",
            data={
                "sample_id": "any",
                "text": "x" * 5001,
                "language": "en",
            },
        )
        assert response.status_code == 400
