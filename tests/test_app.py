from __future__ import annotations

import io
import wave
from dataclasses import dataclass

from fastapi.testclient import TestClient

from asr_service.app import create_app
from asr_service.config import Settings


@dataclass
class FakeWord:
    text: str
    start_time: float
    end_time: float


@dataclass
class FakeTranscription:
    language: str
    text: str
    time_stamps: list[FakeWord] | None = None


class FakeManager:
    def __init__(self) -> None:
        self.is_ready = True
        self.last_transcribe_kwargs: dict[str, object] | None = None

    async def start(self) -> None:
        self.is_ready = True

    async def stop(self) -> None:
        self.is_ready = False

    async def transcribe(self, **kwargs: object) -> list[FakeTranscription]:
        self.last_transcribe_kwargs = kwargs
        return [
            FakeTranscription(
                language="English",
                text="hello world",
                time_stamps=[
                    FakeWord("hello", 0.0, 0.5),
                    FakeWord("world", 0.5, 1.0),
                ],
            )
        ]


def _wav_bytes() -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 16000)
    return buffer.getvalue()


def test_models_endpoint_enforces_bearer_auth() -> None:
    settings = Settings.from_env({"ASR_ENABLED_MODELS": "0.6b", "ASR_API_KEY": "token"})
    client = TestClient(create_app(settings=settings, model_manager=FakeManager()))

    response = client.get("/v1/models")
    assert response.status_code == 401

    response = client.get("/v1/models", headers={"Authorization": "Bearer token"})
    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "qwen3-asr-0.6b"


def test_transcription_verbose_json_returns_words_and_segments() -> None:
    settings = Settings.from_env({"ASR_ENABLED_MODELS": "0.6b", "ASR_ENABLE_ALIGNER": "true"})
    client = TestClient(create_app(settings=settings, model_manager=FakeManager()))

    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sample.wav", _wav_bytes(), "audio/wav")},
        data={
            "model": "qwen3-asr-0.6b",
            "response_format": "verbose_json",
            "timestamp_granularities[]": ["word", "segment"],
            "temperature": "0",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["text"] == "hello world"
    assert payload["language"] == "English"
    assert payload["duration"] == 1.0
    assert payload["words"][0]["word"] == "hello"
    assert payload["segments"][0]["text"] == "hello world"


def test_transcription_rejects_timestamp_requests_when_aligner_disabled() -> None:
    settings = Settings.from_env({"ASR_ENABLED_MODELS": "0.6b", "ASR_ENABLE_ALIGNER": "false"})
    client = TestClient(create_app(settings=settings, model_manager=FakeManager()))

    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sample.wav", _wav_bytes(), "audio/wav")},
        data={
            "model": "qwen3-asr-0.6b",
            "response_format": "srt",
        },
    )

    assert response.status_code == 400
    assert "ASR_ENABLE_ALIGNER=true" in response.json()["detail"]


def test_transcription_normalizes_language_alias() -> None:
    settings = Settings.from_env({"ASR_ENABLED_MODELS": "0.6b"})
    manager = FakeManager()
    client = TestClient(create_app(settings=settings, model_manager=manager))

    response = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("sample.wav", _wav_bytes(), "audio/wav")},
        data={
            "model": "qwen3-asr-0.6b",
            "language": "En",
        },
    )

    assert response.status_code == 200
    assert manager.last_transcribe_kwargs is not None
    assert manager.last_transcribe_kwargs["language"] == "English"
